from Sparse import *
import logging
import torch
import cc3d
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def calculate_eikonal_field(full, border0, border1, fmm = True):
    """
    Calculate bi-Eikonal normalizated coordinate fields on the GPU as described in Biasi et al. (2024).
    
    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    border0 : Sparsevolume
        Sparse volume of the border points where the field is equal to 0.
    border1 : SparseVolume
        Sparse volume of the border points where the field is equal to 1.
    
    Returns
    -------
    SparseField
        Sparse field containing the coordinate values for the given points.
    """

    logger.debug(f"volume: {full.n_points}, border0: {border0.n_points}, border1: {border1.n_points}")

    distances0 = border0.distances_to_points(full.points, fmm=fmm)
    distances1 = border1.distances_to_points(full.points, fmm=fmm)

    field = distances0 / (distances0 + distances1)

    field = SparseField.from_Sparse_with_values(full, field)
    return field

def extract_bridge_path(full, pulmonary, tricuspid, rotation_angle=20.0):
    """
    Extract a path that connects the pulmonary valve to the tricuspid valve, later used to define the intervalvular bridge.

    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full RV.
    pulmonary : SparseVolume
        Sparse volume of the pulmonary valve.
    tricuspid : SparseVolume
        Sparse volume of the tricuspid valve.
    rotation_angle : float, optional
        Angle in degrees to rotate the reference vector between the pulmonary and tricuspid centers, by default 20.0.

    Returns
    -------
    bridge_path : SparseVolume
        Sparse volume containing a path that connects the pulmonary valve to the tricuspid valve.
    """
    starting_point = _get_starting_point_advanced(pulmonary, tricuspid, rotation_angle)

    distance_transform = pulmonary.distances_to_points(full.points, fmm=True)
    distance_transform = SparseField.from_Sparse_with_values(full, distance_transform)
    list_of_bridge_positions = [starting_point]
    distance = distance_transform.values[torch.where(full.check_point_membership(starting_point.reshape(1,-1)))[0][0]].item()
    offsets = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], device=full.points.device)

    while distance > 0:
        candidate_positions = list_of_bridge_positions[-1] + offsets
        candidate_position_object = SparseVolume.from_Sparse_with_points(full, candidate_positions)
        mask_for_candidate_positions = candidate_position_object.check_point_membership(full.points)

        winning_index = torch.argmin(distance_transform.values[mask_for_candidate_positions])
        shortest_distance = distance_transform.values[mask_for_candidate_positions][winning_index]
        list_of_bridge_positions.append(distance_transform.points[mask_for_candidate_positions][winning_index])

        distance = shortest_distance

    bridge_points = torch.vstack(list_of_bridge_positions)
    bridge_path = SparseVolume.from_Sparse_with_points(full, bridge_points)

    return bridge_path

def _get_starting_point(pulmonary, tricuspid):
    """
    Get the starting point for the intervalvular bridge path, which is the closest point on the tricuspid valve to the pulmonary valve.

    Parameters
    ----------
    pulmonary : SparseVolume
        Sparse volume of the pulmonary valve.
    tricuspid : SparseVolume
        Sparse volume of the tricuspid valve.

    Returns
    -------
    torch.Tensor
        The starting point for the intervalvular bridge path.
    """
    pulmonary_center = pulmonary.get_centroid()
    distances_to_center = torch.linalg.norm(tricuspid.points - pulmonary_center, axis=1)
    starting_point = tricuspid.points[distances_to_center.argmin()]
    
    return starting_point

def exchange_non_connected_points(sparse_volume1, sparse_volume2):
    """
    Exchange non-connected points between two sparse volumes to make both volumes fully connected. When I separate two volumes (RV univalve/ RV bridge or two halves of ventricles for rotationals), it happens that the two volumes stop being 6-connected.
    This function finds these points and exchanges them between the two volumes, so that both volumes are fully connected.

    Parameters
    ----------
    sparse_volume1 : SparseVolume
        The first sparse volume.
    sparse_volume2 : SparseVolume
        The second sparse volume.

    Returns
    -------
    SparseVolume, SparseVolume
        The two sparse volumes with non-connected points exchanged.
    """
    full_matrix = sparse_volume1.reconstruct(numpy=True)
    connected_surfaces = cc3d.connected_components(full_matrix, connectivity = 6)
    unique_components, counts = np.unique(connected_surfaces, return_counts=True)

    # Remove component 0 (background) and its count
    mask = unique_components != 0
    unique_components = unique_components[mask]
    counts = counts[mask]

    if len(unique_components) == 1:
        logger.debug("All points in sparse_volume1 are connected.")
    else:
        largest_component = unique_components[np.argmax(counts)]
        points_to_exchange = np.argwhere((connected_surfaces != largest_component) & (connected_surfaces != 0))
        logger.debug(f"Passing {len(points_to_exchange)} points from sparse_volume1 to sparse_volume2.")
        points_to_exchange = SparseVolume.from_Sparse_with_points(sparse_volume2, points_to_exchange)
        where_are_points = points_to_exchange.check_point_membership(sparse_volume1.points)
        sparse_volume1 = sparse_volume1.extract_points_based_on_mask(~where_are_points)
        sparse_volume2 = SparseVolume.combine([sparse_volume2, points_to_exchange])

    full_matrix = sparse_volume2.reconstruct(numpy=True)
    connected_surfaces = cc3d.connected_components(full_matrix, connectivity = 6)
    unique_components, counts = np.unique(connected_surfaces, return_counts=True)

    # Remove component 0 (background) and its count
    mask = unique_components != 0
    unique_components = unique_components[mask]
    counts = counts[mask]

    if len(unique_components) == 1:
        logger.debug("All points in sparse_volume2 are connected.")
    else:
        largest_component = unique_components[np.argmax(counts)]
        points_to_exchange = np.argwhere((connected_surfaces != largest_component) & (connected_surfaces != 0))
        logger.debug(f"Passing {len(points_to_exchange)} points from sparse_volume2 to sparse_volume1.")
        points_to_exchange = SparseVolume.from_Sparse_with_points(sparse_volume1, points_to_exchange)
        where_are_points = points_to_exchange.check_point_membership(sparse_volume2.points)
        sparse_volume2 = sparse_volume2.extract_points_based_on_mask(~where_are_points)
        sparse_volume1 = SparseVolume.combine([sparse_volume1, points_to_exchange])

    return sparse_volume1, sparse_volume2


def _get_starting_point_advanced(pulmonary, tricuspid, rotation_angle=20.0):
    """
    Get the starting point for the intervalvular bridge path, chosen through rotation of a reference vector between the pulmonary and tricuspid centers.
    Inspired by CobivecoX.

    Parameters
    ----------
    pulmonary : SparseVolume
        Sparse volume of the pulmonary valve.
    tricuspid : SparseVolume
        Sparse volume of the tricuspid valve.

    Returns
    -------
    torch.Tensor
        The starting point for the intervalvular bridge path.
    """
    tricuspid_center = tricuspid.get_centroid()
    pulmonary_center = pulmonary.get_centroid()
    tricuspid_to_pulmonary = pulmonary_center - tricuspid_center

    #finding the best fitting plane for the tricuspid valve
    svd = torch.linalg.svd((tricuspid.points - tricuspid_center).T)
    left = svd[0]
    normal = left[:, -1]

    # connect tricuspid to pulmonary and rotate on plane
    pointing_vector = tricuspid_to_pulmonary - torch.dot(tricuspid_to_pulmonary, normal) * normal
    pointing_vector = pointing_vector / torch.linalg.norm(pointing_vector)

    angle_rad = torch.deg2rad(torch.tensor(rotation_angle))
    rotated_vector = (
        pointing_vector * torch.cos(angle_rad)
        + torch.linalg.cross(normal, pointing_vector) * torch.sin(angle_rad)
        + normal * torch.dot(normal, pointing_vector) * (1 - torch.cos(angle_rad))
    )
    rotated_vector = rotated_vector / torch.linalg.norm(rotated_vector)

    vectors_to_points = tricuspid.points - tricuspid_center
    vectors_to_points = vectors_to_points / torch.linalg.norm(vectors_to_points, dim=1, keepdim=True)
    dot_products = torch.sum(vectors_to_points * rotated_vector, dim=1)
    closest_index = torch.argmax(dot_products)
    starting_point = tricuspid.points[closest_index]

    return starting_point



def define_intervalvular_edges( bridge, pulmonary, tricuspid, endo, epi, apex):
    """
    Divides the edge of the intervalvular bridge into 6 faces to be used for UVC calculation in the bridge.

    Parameters
    ----------
    bridge : SparseVolume
        Sparse volume of the intervalvular bridge.
    pulmonary : SparseVolume
        Sparse volume of the pulmonary valve.
    tricuspid : SparseVolume
        Sparse volume of the tricuspid valve.
    endo : SparseVolume
        Sparse volume of the endocardium points.
    epi : SparseVolume
        Sparse volume of the epicardium points.
    apex : SparseVolume
        Sparse volume containing the complete apex landmark points.

    Returns
    -------

    pulmonary_bridge_edge : SparseVolume
        Edge of the bridge on the pulmonary valve.
    tricuspid_bridge_edge : SparseVolume
        Edge of the bridge on the tricuspid valve.
    close_to_septum_edge : SparseVolume
        Edge of the bridge closer to the septum.
    far_from_septum_edge : SparseVolume
        Edge of the bridge farther from the septum.
    """
    
    bridge_edge = bridge.extract_edge_points()
    pulmonary_bridge_edge_mask = pulmonary.check_point_membership(bridge_edge.points)
    pulmonary_bridge_edge = bridge_edge.extract_points_based_on_mask(pulmonary_bridge_edge_mask)
    tricuspid_bridge_edge_mask = tricuspid.check_point_membership(bridge_edge.points)
    tricuspid_bridge_edge = bridge_edge.extract_points_based_on_mask(tricuspid_bridge_edge_mask)
    epi_bridge_edge_mask = epi.check_point_membership(bridge_edge.points)
    epi_bridge_edge = bridge_edge.extract_points_based_on_mask(epi_bridge_edge_mask)
    endo_bridge_edge_mask = endo.check_point_membership(bridge_edge.points)
    endo_bridge_edge = bridge_edge.extract_points_based_on_mask(endo_bridge_edge_mask)
    new_edge_mask = ~(pulmonary_bridge_edge_mask | tricuspid_bridge_edge_mask | epi_bridge_edge_mask | endo_bridge_edge_mask)
    new_edge = bridge_edge.extract_points_based_on_mask(new_edge_mask)

    new_edge_partition = new_edge.extract_connected_surfaces()
    counts = [len(surf.points) for surf in new_edge_partition]
    
    if len(new_edge_partition) != 2:
        logger.warning(f"The new edge should be partitioned into two connected surfaces, but found: {len(new_edge_partition)}, with counts: {counts}")
        logger.warning("Using the two largest connected surfaces.")

    sorted_indices = np.argsort(counts)

    surface1 = new_edge_partition[sorted_indices[-1]]
    surface2 = new_edge_partition[sorted_indices[-2]]

    connector = surface1.get_centroid() - surface2.get_centroid()
    tricuspid_to_pulmonary = pulmonary_bridge_edge.get_centroid() - tricuspid_bridge_edge.get_centroid()
    apicobasal_vector = tricuspid_bridge_edge.get_centroid() - apex.get_centroid()
    test_vector = torch.linalg.cross(connector, tricuspid_to_pulmonary)

    if torch.dot(test_vector, apicobasal_vector) > 0:
        close_to_septum_edge = surface1
        far_from_septum_edge = surface2
    else:
        close_to_septum_edge = surface2
        far_from_septum_edge = surface1

    return pulmonary_bridge_edge, tricuspid_bridge_edge, close_to_septum_edge, far_from_septum_edge

def get_new_apex(full, apex, endo):
    """
    Get the new apex landmark by linking the an apex landmark on the epicardium to the endocardium. This is done by calculating the distance transform from the endocardium and using it to construct the shortest path from the initial apex to the endocardium. 

    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    apex : torch.Tensor
        Initial apex location in the form of a tensor with shape (1, 3) in order (z, y, x).
    endo: SparseVolume
        Sparse volume of the endocardium points.

    Returns
    -------
    SparseVolume
        Sparse volume containing the complete apex landmark points.
    """
    initial_apex = SparseVolume.from_Sparse_with_points(full, apex)

    distance_to_endo = endo.distances_to_points(full.points, fmm=True)
    distance_field = SparseField.from_Sparse_with_values(full, distance_to_endo)

    offsets = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], device=full.points.device)

    list_of_apex_positions = [apex]
    distance = distance_field.values[torch.where(full.check_point_membership(initial_apex.points[0].reshape(1,-1)))[0]]
    while distance > 0:
        candidate_positions = list_of_apex_positions[-1] + offsets
        candidate_position_object = SparseVolume.from_Sparse_with_points(full, candidate_positions)
        mask_for_candidate_positions = candidate_position_object.check_point_membership(distance_field.points)

        winning_index = torch.argmin(distance_field.values[mask_for_candidate_positions])
        shortest_distance = distance_field.values[mask_for_candidate_positions][winning_index]
        list_of_apex_positions.append(distance_field.points[mask_for_candidate_positions][winning_index])

        distance = shortest_distance
    

    apex_points = torch.vstack(list_of_apex_positions)
    complete_apex = SparseVolume.from_Sparse_with_points(full, apex_points)

    return complete_apex




def calculate_rotational_dirty(full, apex, rotational_reference, base):
    """
    Calculate the rotational coordinates on a ventricular mesh. the axis of orientation is from the apex to the base. 
    
    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    apex: torch.Tensor
        Apex location in the form of a tensor with shape (3,) in order (z, y, x).
    rotational_reference : torch.Tensor
        Reference point for the rotational coordinates, in the form of a tensor with shape (3,) in order (z, y, x).
    base : SparseVolume
        Sparse volume of the base points.
    
    Returns
    -------
    SparseField
        Sparse field containing the rotational coordinates.
    """

    centre = base.get_centroid()

    short_axis = apex - centre
    short_axis = short_axis / torch.linalg.norm(short_axis)

    ref = rotational_reference
    centre_to_reference = ref - centre
    centre_to_reference = centre_to_reference / torch.linalg.norm(centre_to_reference)

    rotational_vector = centre_to_reference - torch.dot(centre_to_reference, short_axis) * short_axis
    rotational_vector = rotational_vector / torch.linalg.norm(rotational_vector)

    centre_to_points = full.points - centre
    centre_to_points = centre_to_points / torch.linalg.norm(centre_to_points, dim=1, keepdim=True)
    projected_vectors = centre_to_points - torch.sum(centre_to_points * short_axis, dim=1).reshape((-1,1)) * short_axis
    projected_vectors = projected_vectors / torch.linalg.norm(projected_vectors, dim=1, keepdim=True)
    cross = torch.cross(projected_vectors, rotational_vector.reshape(1,-1), dim=1)
    indicator = torch.sum(cross * short_axis, dim=1) > 0

    angles = torch.acos(torch.sum(projected_vectors * rotational_vector, dim=1))
    angles[indicator]= 2*torch.pi - angles[indicator]

    rotational = SparseField.from_Sparse_with_values(full, angles)
    return rotational

def calculate_rotational_advanced(full, border, apex, rotational_reference, centre, smooth_iter = 0, smooth_lambda_param=0.99, reverse=False):
    """
    Calculate the rotational coordinates on a ventricular mesh. the axis of orientation is from the apex to the base. 
    
    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    border : SparseField
        Sparse field of the border points.
    apex: torch.Tensor
        Apex location in the form of a tensor with shape (3,) in order (z, y, x).
    rotational_reference : torch.Tensor
        Reference point for the rotational coordinates, in the form of a tensor with shape (3,) in order (z, y, x).
    base : SparseVolume
        Sparse volume of the base points.
    centre : torch.Tensor
        Centre of the base in the form of a tensor with shape (3,) in order (z, y, x).
    smooth_iter : int, optional
        Number of smoothing iterations to apply to the rotational field, by default 0.
    smooth_lambda_param : float, optional
        Lambda parameter for the smoothing function, by default 0.99.
    reverse : bool, optional
        If True, the front and back points are swapped, by default False.

    Returns
    -------
    SparseField
        Sparse field containing the rotational coordinates.
    """
    front, back = _split_into_front_and_back(full, apex, rotational_reference, centre, reverse=reverse)
    front, back = exchange_non_connected_points(front, back)

    front_field = _calculate_rotational_on_half_ventricle(front, border, apex, rotational_reference, centre)
    back_field = _calculate_rotational_on_half_ventricle(back, border, apex, rotational_reference, centre)

    if smooth_iter > 0:
        #I smooth here and not after joining because we can't smooth where there is a discontinuity in the values.
        front_field = front_field.forward_smooth(lambda_param=smooth_lambda_param, iter=smooth_iter, values_to_keep=[0, 1])
        back_field = back_field.forward_smooth(lambda_param=smooth_lambda_param, iter=smooth_iter, values_to_keep=[0, 1])

    front_field = SparseField.from_Sparse_with_values(front, 0.5*front_field.values)
    back_field = SparseField.from_Sparse_with_values(back, 1 - 0.5*back_field.values)

    rotational = SparseField.combine([front_field, back_field])
    rotational = rotational.reorder(full)
    rotational.replace_nan()

    return rotational


def _split_into_front_and_back(full, apex, rotational_reference, centre, reverse=False):
    """
    Split the full volume into front and back through a plane defined by the apex, the rotational reference and the centre of the base.

    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    apex : torch.Tensor
        Apex location in the form of a tensor with shape (3,) in order (z, y, x).
    rotational_reference : torch.Tensor
        Reference point for the rotational coordinates, in the form of a tensor with shape (3,) in order (z, y, x).   
    centre : torch.Tensor
        Centre of the base in the form of a tensor with shape (3,) in order (z, y, x).
    reverse : bool, optional
        If True, the front and back points are swapped, by default False.

    Returns
    -------
    front : SparseVolume
        Sparse volume of the front points.
    back : SparseVolume
        Sparse volume of the back points.
    """
    short_axis = apex - centre
    centre_to_reference = rotational_reference - centre
    rotational_vector = centre_to_reference - torch.dot(centre_to_reference, short_axis) * short_axis

    centre_to_points = full.points - centre
    cross = torch.cross(rotational_vector.reshape(1, -1), centre_to_points, dim=1)

    front_mask = torch.sum(cross * short_axis, dim=1) > 0

    if reverse:
        front_mask = ~front_mask

    front = full.extract_points_based_on_mask(front_mask)
    back = full.extract_points_based_on_mask(~front_mask)

    return front, back

def _split_into_front_and_back_septal(full, apex, septal_center, septal_normal, reverse=False):
    """
    Split the full volume into front and back through a plane defined by the apex, the rotational reference and the centre of the base.

    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    apex : torch.Tensor
        Apex location in the form of a tensor with shape (3,) in order (z, y, x).
    rotational_reference : torch.Tensor
        Reference point for the rotational coordinates, in the form of a tensor with shape (3,) in order (z, y, x).   
    centre : torch.Tensor
        Centre of the base in the form of a tensor with shape (3,) in order (z, y, x).
    reverse : bool, optional
        If True, the front and back points are swapped, by default False.

    Returns
    -------
    front : SparseVolume
        Sparse volume of the front points.
    back : SparseVolume
        Sparse volume of the back points.
    """
    short_axis = septal_center - apex

    centre_to_points = full.points - septal_center
    cross = torch.cross(septal_normal.reshape(1, -1), centre_to_points, dim=1)

    front_mask = torch.sum(cross * short_axis, dim=1) > 0

    if reverse:
        front_mask = ~front_mask

    front = full.extract_points_based_on_mask(front_mask)
    back = full.extract_points_based_on_mask(~front_mask)

    return front, back

def _calculate_rotational_on_half_ventricle(half, border, apex, rotational_reference, centre):
    """
    Calculate the rotational coordinates on one half of the ventricle after splitting it into front and back. 

    Parameters
    ----------
    half : SparseVolume
        Sparse volume of the half ventricle (either front or back).
    border : SparseField
        Sparse field of the border points of the entire ventricle.
    apex : torch.Tensor
        Apex location in the form of a tensor with shape (3,) in order (z, y, x).
    rotational_reference : torch.Tensor
        Reference point for the rotational coordinates, in the form of a tensor with shape (3,) in order (z, y, x).
    centre : torch.Tensor
        Centre of the base in the form of a tensor with shape (3,) in order (z, y, x).

    Returns
    -------
    SparseField
        Sparse field containing the rotational coordinates for one half of the ventricle.
    """
    short_axis = apex - centre
    centre_to_reference = rotational_reference - centre

    half_border = half.extract_edge_points()
    new_points_mask = border.check_point_membership(half_border.points)
    half_border = half_border.extract_points_based_on_mask(~new_points_mask)
    centre_to_border_points = half_border.points - centre
    cross = torch.cross(short_axis.reshape(1, -1), centre_to_border_points, dim=1)
    closer_evaluator = torch.linalg.cross(short_axis, centre_to_reference)
    closer_to_ref = torch.sum(cross * closer_evaluator, dim=1) > 0

    border_closer_to_ref = half_border.extract_points_based_on_mask(closer_to_ref)
    border_farther_from_ref = half_border.extract_points_based_on_mask(~closer_to_ref)
    field_half = calculate_eikonal_field(half, border0=border_closer_to_ref, border1=border_farther_from_ref)
    return field_half

def calculate_rotational_cobiveco(free_volume, septum_volume, border, initial_apex, ventricle_center, base_center, iter_smooth_rot, lambda_param, reverse=False):
    """
    Calculate the rotational coordinates on a ventricular mesh using the Cobiveco method.

    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    border : SparseField
        Sparse field of the border points with the septal labels.
    transventricular : SparseVolume
        Sparse field of the transventricular points.
    lv_apex : torch.Tensor
        Apex location of the left ventricle in the form of a tensor with shape (3,) in order (z, y, x).
    rv_apex : torch.Tensor
        Apex location of the right ventricle in the form of a tensor with shape (3,) in order (z, y, x).
    base : SparseVolume
        Sparse volume of the base points.
    """

    #calculating in free volume
    free_volume_edge = free_volume.extract_edge_points()
    new_points_mask = border.check_point_membership(free_volume_edge.points)
    new_edge = free_volume_edge.extract_points_based_on_mask(~new_points_mask)

    apex_to_new_edge_points = (new_edge.points - initial_apex).to(dtype=torch.float32)
    apex_to_ventricle_center = ventricle_center - initial_apex
    apex_to_base_center = base_center - initial_apex

    cross = torch.cross(apex_to_new_edge_points, apex_to_ventricle_center.reshape(1, -1), dim=1)
    dot = torch.sum(cross * apex_to_base_center, dim=1)
    mask = dot > 0
    if reverse:
        mask = ~mask
    anterior_edge = new_edge.extract_points_based_on_mask(mask)
    posterior_edge = new_edge.extract_points_based_on_mask(~mask)

    free_volume_field = calculate_eikonal_field(free_volume, border0=anterior_edge, border1=posterior_edge)

    #calculating in septal volume
    septum_edge = septum_volume.extract_edge_points()
    new_points_mask = border.check_point_membership(septum_edge.points)
    new_edge = septum_edge.extract_points_based_on_mask(~new_points_mask)

    apex_to_new_edge_points = (new_edge.points - initial_apex).to(dtype=torch.float32)
    apex_to_ventricle_center = ventricle_center - initial_apex
    apex_to_base_center = base_center - initial_apex

    cross = torch.cross(apex_to_new_edge_points, apex_to_ventricle_center.reshape(1, -1), dim=1)
    dot = torch.sum(cross * apex_to_base_center, dim=1)
    mask = dot > 0
    if reverse:
        mask = ~mask
    anterior_edge = new_edge.extract_points_based_on_mask(mask)
    posterior_edge = new_edge.extract_points_based_on_mask(~mask)

    septum_field = calculate_eikonal_field(septum_volume, border0=anterior_edge, border1=posterior_edge)

    #smoothing the fields
    if iter_smooth_rot > 0:
        free_volume_field = free_volume_field.forward_smooth(lambda_param=lambda_param, iter=iter_smooth_rot, values_to_keep=[0, 1])
        septum_field = septum_field.forward_smooth(lambda_param=lambda_param, iter=iter_smooth_rot, values_to_keep=[0, 1])

    #combining the two fields
    free_volume_field_values = free_volume_field.values *(2/3)
    septum_field_values =  1- (septum_field.values)* (1/3)

    combined_field = SparseField.combine([SparseField.from_Sparse_with_values(free_volume, free_volume_field_values),
                                          SparseField.from_Sparse_with_values(septum_volume, septum_field_values)])
    
    return combined_field


def split_base_into_pulmonary_and_tricuspid_biventricular(rv, rv_base, rv_apex, rv_septal_surface):
    """
    Split the base of the right ventricle into the pulmonary and tricuspid valves using a chirality argument.

    Parameters
    ----------
    rv : SparseVolume
        Sparse volume of the right ventricle.
    rv_base : SparseVolume
        Sparse volume of the base of the right ventricle.
    rv_apex : torch.Tensor
        Apex location of the right ventricle in the form of a tensor with shape (3,) in order (z, y, x).
    rv_septal_surface : SparseVolume
        Sparse volume of the septal surface of the right ventricle.

    Returns
    -------
    pulmonary_valve : SparseVolume
        Sparse volume of the pulmonary valve.
    tricuspid_valve : SparseVolume
        Sparse volume of the tricuspid valve.
    """

    base_matrix = rv_base.reconstruct(numpy=True)
    connected_surfaces = cc3d.connected_components(base_matrix, connectivity=26)
    unique_components, counts = np.unique(connected_surfaces, return_counts=True)

    # Remove component 0 (background) and its count
    mask = unique_components != 0
    unique_components = unique_components[mask]
    counts = counts[mask]

    if len(unique_components)!= 2:
        logger.warning(f"The base of the right ventricle should be partitioned into two connected surfaces, but found: {len(unique_components)} with counts {counts}.")
        logger.warning("Using the two largest connected surfaces.")

    sorted_indices = np.argsort(counts)

    #two valves we test are the two connected surfaces with the largest number of points
    valve1 = np.argwhere(connected_surfaces == unique_components[sorted_indices[-1]])
    valve2 = np.argwhere(connected_surfaces == unique_components[sorted_indices[-2]])

    valve1 = SparseVolume.from_Sparse_with_points(rv_base, valve1)
    valve2 = SparseVolume.from_Sparse_with_points(rv_base, valve2)


    valve1_center = valve1.get_centroid()
    valve2_center = valve2.get_centroid()
    bivalve_center = (valve1_center + valve2_center) / 2

    bivalve_center_to_valve1 = valve1_center - bivalve_center
    bivalve_center_to_valve2 = valve2_center - bivalve_center

    apex_to_bivalve_center = bivalve_center - rv_apex
    cross_valve1 = torch.linalg.cross(apex_to_bivalve_center, bivalve_center_to_valve1)
    cross_valve2 = torch.linalg.cross(apex_to_bivalve_center, bivalve_center_to_valve2)

    rv_center = rv.get_centroid()
    septal_center = rv_septal_surface.get_centroid()
    rv_center_to_septal = septal_center - rv_center

    dot_valve1 = torch.dot(cross_valve1, rv_center_to_septal)
    dot_valve2 = torch.dot(cross_valve2, rv_center_to_septal)

    #imagine a cross product between the vertical vector going from the apex to the base and
    #the vector going from the middle of rv to the pulmonary valve. we expect the cross product to be 
    #oriented from the center of the rv to the septum, aka the dot product of that cross product with
    #the vector going from the center of the rv to the septum should be positive.
    #similarly, the cross product with the tricuspid valve is oriented towards the free wall.

    #This whole chirality argument is valid when im dealing with (x,y,z), but if I switch to (z,y,x) like here, argument is inverted.

    if dot_valve1 < dot_valve2:
        pulmonary_valve = valve1
        tricuspid_valve = valve2
    else:
        pulmonary_valve = valve2
        tricuspid_valve = valve1

    return pulmonary_valve, tricuspid_valve


def find_initial_apex_univentricular(full, base, epi):
    """
    Find the initial apex of a univentricular volume as the epicardium point farthest from the base.

    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    base : SparseVolume
        Sparse volume of the base points.
    epi : SparseVolume
        Sparse volume of the epicardium points.

    Returns
    -------
    torch.Tensor
        The initial apex location in the form of a tensor with shape (1,3) in order (z, y, x).
    """

    base_transform = base.distances_to_points(full.points, fmm=True)
    base_transform = SparseField.from_Sparse_with_values(full, base_transform)
    transform_on_epi = base_transform.values_for_closest_points(epi)

    max_index = torch.argmax(transform_on_epi.values)
    initial_apex = epi.points[max_index].reshape(1, -1)
    return initial_apex

def find_initial_apex_biventricular(full, full_base, lv_septal_surface, rv_septal_surface):
    """
    Find the initial apex of a biventricular volume as the epicardium point farthest from the base.

    Parameters
    ----------
    full : SparseVolume
        Sparse volume of the full image.
    full_base : SparseVolume
        Sparse volume of the base points.
    lv_septal_surface : SparseVolume
        Sparse volume of the septal surface of the left ventricle.
    rv_septal_surface : SparseVolume
        Sparse volume of the septal surface of the right ventricle.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        The initial apex locations for the left and right ventricles in the form of tensors with shape (1,3) in order (z, y, x).
    """

    base_transform = full_base.distances_to_points(full.points, fmm=True)
    base_transform = SparseField.from_Sparse_with_values(full, base_transform)
    lv_transform_on_septum = base_transform.values_for_closest_points(lv_septal_surface)
    rv_transform_on_septum = base_transform.values_for_closest_points(rv_septal_surface)
    lv_max_index = torch.argmax(lv_transform_on_septum.values)
    rv_max_index = torch.argmax(rv_transform_on_septum.values)
    lv_initial_apex = lv_septal_surface.points[lv_max_index].reshape(1, -1)
    rv_initial_apex = rv_septal_surface.points[rv_max_index].reshape(1, -1)

    return lv_initial_apex, rv_initial_apex