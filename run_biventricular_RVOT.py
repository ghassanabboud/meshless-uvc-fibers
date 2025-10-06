import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import pickle 
import os
from fibers import *
from uvc import *
import SimpleITK as sitk
import numpy as np
from scipy.spatial import KDTree
import torch
from Sparse import SparseVolume, SparseField, SparseVectors
import h5py
import pyvista as pv

# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path= "configs", config_name="ventricles_config")
def run_pipeline(cfg: DictConfig) -> None:

    start = time.perf_counter()
    logger.info("================ Starting Pipeline For Biventricular Fibers ================")
    results_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    main_name = results_dir.split("/")[-1]

    uvc_dir = results_dir + "/uvc/"
    os.makedirs(uvc_dir, exist_ok=True)

    if cfg.debug:
        intermediate_dir = results_dir + "/intermediate_results/"
        os.makedirs(intermediate_dir, exist_ok=True)
        logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode activated")


    device = cfg.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("cuda is not available. Switching to cpu.")
            device = "cpu"
        else:
            logger.info(f"Using {device} for calculations.")
    elif device.startswith("cpu"):
        logger.info("Using cpu for calculations.")
    else:
        raise ValueError(f"Unknown device: {device}. Choose from 'cuda' or 'cpu'.")
    
    full = SparseVolume.from_mha_file(cfg.files.volume_path, device=device)
    initial_border = SparseField.from_mha_file(cfg.files.border_path, device=device)

    #step 1: Pre-processing
    logger.info("Checking that all points in volume are 6-connected")
    full = full.eliminate_non_connected_points()
    full.to_pickle(results_dir + "/full_volume.pkl")

    logger.info("Extracting and saving new border")
    new_border = full.extract_edge_points()
    new_border = initial_border.values_for_closest_points(new_border)

    #different boundary conditions
    rv_endo = new_border.extract_isosurface(cfg.labels.RV_endo_label)
    lv_endo = new_border.extract_isosurface(cfg.labels.LV_endo_label)
    epi = new_border.extract_isosurface(cfg.labels.epi_label)
    full_base = new_border.extract_isosurface(cfg.labels.base_label)
    full_base_center = full_base.get_centroid()

    #step 2: transventricular coordinate and LV-RV separation
    logger.info("Calculating tranventricular coordinates")
    transventricular = calculate_eikonal_field(full, border0=lv_endo, border1=rv_endo)
    new_values = transventricular.values.clone()
    new_values[new_values < 0.5] = 0
    new_values[new_values >= 0.5] = 1
    transventricular = SparseField.from_Sparse_with_values(transventricular, new_values)

    lv =  transventricular.extract_isosurface(0)
    rv =  transventricular.extract_isosurface(1)
    lv, rv = exchange_non_connected_points(lv, rv)
    transventricular_lv = SparseField.from_Sparse_with_values(lv, torch.zeros(len(lv.points)))
    transventricular_rv = SparseField.from_Sparse_with_values(rv, torch.ones(len(rv.points)))
    transventricular = SparseField.combine([transventricular_lv, transventricular_rv])
    transventricular = transventricular.reorder(full)

    transventricular.to_pickle(uvc_dir + "transventricular.pkl")
    #step 3: defining septal surfaces, and septal apices
    logger.info("Extracting septal surfaces")
    lv_border = lv.extract_edge_points()
    closest_values = new_border.values_for_closest_points(lv_border).values
    mask =  new_border.check_point_membership(lv_border.points)
    closest_values[~mask] = cfg.labels.septal_lv_label
    lv_border = SparseField.from_Sparse_with_values(lv_border, closest_values)

    rv_border = rv.extract_edge_points()
    closest_values = new_border.values_for_closest_points(rv_border).values
    mask =  new_border.check_point_membership(rv_border.points)
    closest_values[~mask] = cfg.labels.septal_rv_label
    rv_border = SparseField.from_Sparse_with_values(rv_border, closest_values)

    final_border = SparseField.combine([lv_border, rv_border])
    final_border.to_pickle(results_dir + "/final_border.pkl")

    # this is for the complete rv and lv before we remove the rv bridge
    logger.info("Extracting new border conditions for LV and RV")
    lv_free_surface = lv_border.extract_isosurface(cfg.labels.epi_label)
    lv_septal_surface = lv_border.extract_isosurface(cfg.labels.septal_lv_label)
    lv_epi_septum = SparseVolume.combine([lv_free_surface, lv_septal_surface])
    lv_base = lv_border.extract_isosurface(cfg.labels.base_label)
    lv_center = lv.get_centroid()

    rv_free_surface = rv_border.extract_isosurface(cfg.labels.epi_label)
    rv_septal_surface = rv_border.extract_isosurface(cfg.labels.septal_rv_label)
    rv_epi_septum = SparseVolume.combine([rv_free_surface, rv_septal_surface])
    rv_base = rv_border.extract_isosurface(cfg.labels.base_label)
    rv_center = rv.get_centroid()

    if cfg.files.apex_path is None:
        logger.info("Apex path is not provided. Switching to automatic apex detection.")
        lv_initial_apex, rv_initial_apex = find_initial_apex_biventricular(full, full_base, lv_septal_surface, rv_septal_surface)
    else:
        lv_initial_apex = lv_septal_surface.closest_indices_from_fcsv(cfg.files.apex_path)
        rv_initial_apex = rv_septal_surface.closest_indices_from_fcsv(cfg.files.apex_path)

    logger.info("Linking the apex to endocardium on each ventricle")
    lv_complete_apex = get_new_apex(lv, lv_initial_apex, lv_endo)
    rv_complete_apex = get_new_apex(rv, rv_initial_apex, rv_endo)

    #step 4: extracting the RV intervalvular bridge and defining its edges
    logger.info("Extracting RV intervalvular bridge")
    pulmonary, tricuspid = split_base_into_pulmonary_and_tricuspid_biventricular(rv, rv_base, rv_initial_apex[0], rv_septal_surface)
    bridge_path= extract_bridge_path(rv, pulmonary, tricuspid, rotation_angle=cfg.bridge_rotation_angle)
    intervalvular_coord = calculate_eikonal_field(rv, border0=bridge_path, border1=rv_complete_apex)
    mask = (intervalvular_coord.values < cfg.intervalvular_threshold) & ~rv_septal_surface.check_point_membership(intervalvular_coord.points)
    bridge = rv.extract_points_based_on_mask(mask)
    univalve = rv.extract_points_based_on_mask(~mask)
    bridge,univalve = exchange_non_connected_points(bridge, univalve)

    if cfg.debug:
        bridge_path.to_pickle(intermediate_dir + "bridge_path.pkl")
        bridge.to_pickle(intermediate_dir + "bridge.pkl")
        intervalvular_coord.to_pickle(intermediate_dir + "intervalvular_coord.pkl")

    # this is for the univalvular volume we get when we cut out the bridge from the RV
    logger.info("Defining RV intervalvular and univalve edges")
    pulmonary_bridge_edge, tricuspid_bridge_edge, close_to_septum_edge, far_from_septum_edge = define_intervalvular_edges(bridge, pulmonary, tricuspid, rv_endo, rv_epi_septum, rv_complete_apex)
    univalve_edge = univalve.extract_edge_points()
    new_points_on_univalve_edge_mask = ~(rv_border.check_point_membership(univalve_edge.points))
    new_univalve_edge_labels = rv_border.values_for_closest_points(univalve_edge).values
    new_univalve_edge_labels[new_points_on_univalve_edge_mask] = cfg.labels.base_label
    univalve_edge = SparseField.from_Sparse_with_values(univalve_edge, new_univalve_edge_labels)
    univalve_base = univalve_edge.extract_isosurface(cfg.labels.base_label)
    univalve_septal_surface = univalve_edge.extract_isosurface(cfg.labels.septal_rv_label)
    univalve_epi = univalve_edge.extract_isosurface(cfg.labels.epi_label)

    #step 5: apicobasal coordinates
    logger.info("Calculating apicobasal coordinates for LV and univalvular RV")
    apicobasal_univalve = calculate_eikonal_field(univalve, border0=rv_complete_apex, border1=univalve_base)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        apicobasal_univalve = apicobasal_univalve.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
    apicobasal_lv = calculate_eikonal_field(lv, border0=lv_complete_apex, border1=lv_base)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        apicobasal_lv = apicobasal_lv.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])

    logger.info("Calculating apicobasal coordinates for the RV bridge")
    apicobasal_bridge = calculate_eikonal_field(bridge, border0=far_from_septum_edge, border1=close_to_septum_edge)
    apicobasal_bridge.values = 0.25 * apicobasal_bridge.values + 1  #range [1, 1.25]
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        apicobasal_bridge = apicobasal_bridge.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[1, 1.25])

    apicobasal = SparseField.combine([apicobasal_lv, apicobasal_univalve, apicobasal_bridge])
    apicobasal = apicobasal.reorder(full)

    #step 6: calculating ridge coordinates and separating volumes into free wall and septum for LV and univalvular RV
    logger.info("Calculating ridge coordinates and separating volumes into free wall and septum for LV and univalvular RV")
    univalvular_biventricular = SparseVolume.combine([univalve, lv])
    univalvular_biventricular_epi = SparseVolume.combine([univalve_epi, lv_free_surface])
    univalvular_biventricular_septal_surface = SparseVolume.combine([univalve_septal_surface, lv_septal_surface])

    #we calculate the ridge not by using the entire epi but by using the non-basal epi, this is to have a septal volume that goes up and separates the free walls
    #similar to definition of "non-basal epicardium" in CobivecoX fig.1
    apicobasal_on_epi_roi = apicobasal.values_for_closest_points(univalvular_biventricular_epi)
    nonbasal_epi = univalvular_biventricular_epi.extract_points_based_on_mask(apicobasal_on_epi_roi.values < 0.9)
    ridge_coord = calculate_eikonal_field(univalvular_biventricular, border0=nonbasal_epi, border1=univalvular_biventricular_septal_surface)

    transventricular_indicator = transventricular.values_for_closest_points(univalvular_biventricular)
    rv_free_volume = univalvular_biventricular.extract_points_based_on_mask((transventricular_indicator.values ==1) & (ridge_coord.values <= 0.5))
    rv_septum_volume = univalvular_biventricular.extract_points_based_on_mask((transventricular_indicator.values ==1) & (ridge_coord.values > 0.5))
    rv_free_volume, rv_septum_volume = exchange_non_connected_points(rv_free_volume, rv_septum_volume)

    lv_free_volume = univalvular_biventricular.extract_points_based_on_mask((transventricular_indicator.values ==0) & (ridge_coord.values <= 0.5))
    lv_septum_volume = univalvular_biventricular.extract_points_based_on_mask((transventricular_indicator.values ==0) & (ridge_coord.values > 0.5))
    lv_free_volume, lv_septum_volume = exchange_non_connected_points(lv_free_volume, lv_septum_volume)

    #step 7: rotational coordinates
    logger.info("Calculating rotational coordinates for LV and univalvular RV")
    iter_smooth_rot = cfg.smooth.iter if cfg.smooth.use else 0
    lambda_param = cfg.smooth.lambda_param if cfg.smooth.use else 0.0
    rotational_rv = calculate_rotational_cobiveco( rv_free_volume, rv_septum_volume, univalve_edge, rv_initial_apex[0], rv_center, full_base_center, iter_smooth_rot, lambda_param, reverse=True)
    rotational_lv = calculate_rotational_cobiveco( lv_free_volume, lv_septum_volume, lv_border, lv_initial_apex[0], lv_center, full_base_center, iter_smooth_rot, lambda_param)

    logger.info("Calculating rotational coordinates for the RV bridge")
    rotational_bridge = calculate_eikonal_field(bridge, border0 = pulmonary_bridge_edge, border1=tricuspid_bridge_edge)
    rotational_bridge_new_values = rotational_bridge.values * 0.25 + 1  # range [1, 1.25]
    rotational_bridge = SparseField.from_Sparse_with_values(bridge, rotational_bridge_new_values)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        rotational_bridge = rotational_bridge.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[1, 1.25])

    logger.info("Joining rotational coordinates")
    rotational = SparseField.combine([rotational_lv,rotational_rv, rotational_bridge])
    rotational = rotational.reorder(full)

    #step 8: transmural coordinates
    logger.info("Calculating transmural coordinates for RV and LV")
    transmural_rv = calculate_eikonal_field(rv, border0=rv_endo, border1=rv_epi_septum)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        transmural_rv = transmural_rv.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
    transmural_lv = calculate_eikonal_field(lv, border0=lv_endo, border1=lv_epi_septum)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        transmural_lv = transmural_lv.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
    transmural = SparseField.combine([transmural_lv, transmural_rv])
    transmural = transmural.reorder(full)


    apicobasal.to_pickle(uvc_dir + "apicobasal.pkl")
    transmural.to_pickle(uvc_dir + "transmural.pkl")
    rotational.to_pickle(uvc_dir + "rotational.pkl")

    if cfg.debug:
        ridge_coord.to_pickle(intermediate_dir + "ridge_coord.pkl")
        pulmonary.to_pickle(intermediate_dir + "pulmonary.pkl")
        tricuspid.to_pickle(intermediate_dir + "tricuspid.pkl")
        univalve.to_pickle(intermediate_dir + "univalve.pkl")

    if not cfg.calculate_fiber:
        logger.info("Skipping fiber calculation as per configuration")
        logger.info(f"Pipeline runtime: {time.perf_counter() - start:.2f} seconds")
        return
    
    #step 9: gradient fields of coordinates, special treatment for RV following Doste methodology

    logger.info("Calculating transmural vectors")
    #to match Doste, transmural vectors go out of LV endo and into RV endo
    transmural_vectors_lv = transmural_lv.gradient()
    transmural_rv_to_derive_values = 1 - transmural_rv.values
    transmural_rv_to_derive = SparseField.from_Sparse_with_values(transmural_rv, transmural_rv_to_derive_values)
    transmural_vectors_rv = transmural_rv_to_derive.gradient()

    total_transmural_vectors = SparseVectors.combine([transmural_vectors_lv, transmural_vectors_rv])
    total_transmural_vectors = total_transmural_vectors.reorder(full)

    logger.info("Calculating apicobasal vectors for LV")
    #to match Doste, apicobasal vectors go down from LV base to LV apex
    apicobasal_lv_to_derive_values = 1- apicobasal_lv.values
    apicobasal_lv_to_derive = SparseField.from_Sparse_with_values(apicobasal_lv, apicobasal_lv_to_derive_values)
    apicobasal_vectors_lv = apicobasal_lv_to_derive.gradient()

    logger.info("Calculating apicobasal vectors for RV using Doste methodology")
    logger.info("calculating apex_to_OT, apex_to_base and w_function")
    apex_to_OT = calculate_eikonal_field(rv, border0=pulmonary, border1=rv_complete_apex)
    apex_to_base = calculate_eikonal_field(rv, border0=tricuspid, border1=rv_complete_apex)
    w_function_original = calculate_eikonal_field(rv, border0=pulmonary, border1=SparseVolume.combine([tricuspid, rv_complete_apex]))
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        apex_to_OT = apex_to_OT.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
        apex_to_base = apex_to_base.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
        w_function_original = w_function_original.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])

    logger.info(f"Adjusting w function using sigmoid function with alpha={cfg.doste.theta} and x0={cfg.doste.x0}")
    sigmoid = lambda x, theta, x0: 1 / (1 + torch.exp(-theta * (x-x0)))
    w_values = sigmoid(w_function_original.values, cfg.doste.theta, cfg.doste.x0)
    w_function = SparseField.from_Sparse_with_values(w_function_original, w_values)

    logger.info("Calculating apicobasal basal vectors")
    basal_vectors_sparse = apex_to_base.gradient()

    logger.info("Calculating apicobasal outflow tract vectors")
    OT_vectors_sparse = apex_to_OT.gradient()

    logger.info("interpolating apicobasal vectors using the w function")
    w_values = w_function.values
    basal_vectors = basal_vectors_sparse.vectors
    OT_vectors = OT_vectors_sparse.vectors

    apicobasal_vectors_rv = w_values[..., None] * basal_vectors + (1 - w_values[..., None]) * OT_vectors
    norms = torch.linalg.norm(apicobasal_vectors_rv, dim=1)
    norms[norms == 0] = 1
    apicobasal_vectors_rv = apicobasal_vectors_rv / norms[..., None]
    apicobasal_vectors_rv = SparseVectors.from_Sparse_with_vectors(basal_vectors_sparse, apicobasal_vectors_rv)

    total_apicobasal_vectors = SparseVectors.combine([apicobasal_vectors_lv, apicobasal_vectors_rv])
    total_apicobasal_vectors = total_apicobasal_vectors.reorder(full)

    #step 10: alpha and beta values
    logger.info("Calculating alpha and beta values")

    alpha_lv = get_alpha_values(transmural_lv, alpha_endo=cfg.angles.alpha_LV_endo, alpha_epi=cfg.angles.alpha_LV_epi)
    alpha_rv = get_alpha_values_doste(transmural_rv, w_function, alpha_endo=cfg.angles.alpha_RV_endo, alpha_epi=cfg.angles.alpha_RV_epi, alpha_OT_endo=cfg.angles.alpha_OT_endo, alpha_OT_epi=cfg.angles.alpha_OT_epi)
    alpha = SparseField.combine([alpha_lv, alpha_rv])
    alpha = alpha.reorder(full)
    beta = None
    
    if cfg.calculate_sheet or cfg.calculate_transverse:
        beta_lv = get_beta_values(transmural_lv, beta_endo=cfg.angles.beta_LV_endo, beta_epi=cfg.angles.beta_LV_epi)
        beta_rv = get_beta_values_doste(transmural_rv, w_function, beta_endo=cfg.angles.beta_RV_endo, beta_epi=cfg.angles.beta_RV_epi)
        beta = SparseField.combine([beta_lv, beta_rv])
        beta = beta.reorder(full)

    #step 11: septal coordinates for blending alpha values in the septum (optional)
    if cfg.blend_at_septum:
        logger.info("Calculating septal coordinates for blending alpha values in the septum")
        septal_coord = calculate_eikonal_field(full, border0=SparseVolume.combine([rv_endo, lv_endo]), border1=SparseVolume.combine([rv_septal_surface, lv_septal_surface]))
        if cfg.smooth.use:
            logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
            septal_coord = septal_coord.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
        alpha_values = alpha.values * (1 - septal_coord.values) + cfg.angles.alpha_septal * septal_coord.values
        alpha = SparseField.from_Sparse_with_values(alpha, alpha_values)

        if cfg.calculate_sheet or cfg.calculate_transverse:
            beta_values = beta.values * (1 - septal_coord.values) + cfg.angles.beta_septal * septal_coord.values
            beta = SparseField.from_Sparse_with_values(beta, beta_values)

    #step 12: local coordinate system
    logger.info("Calculating local coordinate system")
    e_l, e_t, e_c = get_local_system(total_apicobasal_vectors, total_transmural_vectors)

    #step 13: fiber, sheet and transverse vectors through rotations
    logger.info("Calculating rotations for fiber, sheet and transverse vectors")
    fiber, sheet, transverse = calculate_fibers(e_c, e_l, e_t, alpha, beta, calculate_sheet=cfg.calculate_sheet, calculate_transverse=cfg.calculate_transverse)
    fiber.interpolate_missing_vectors()
    fiber.to_pickle(results_dir + "/fiber.pkl")

    if cfg.calculate_sheet:
        sheet.interpolate_missing_vectors()
        sheet.to_pickle(results_dir + "/sheet.pkl")
    if cfg.calculate_transverse:
        transverse.interpolate_missing_vectors()
        transverse.to_pickle(results_dir + "/transverse.pkl")

    if cfg.debug:
        e_l.to_pickle(intermediate_dir + "e_l.pkl")
        e_c.to_pickle(intermediate_dir + "e_c.pkl")
        e_t.to_pickle(intermediate_dir + "e_t.pkl")
        alpha.to_pickle(intermediate_dir + "alpha.pkl")
        w_function.to_pickle(intermediate_dir + "w_function.pkl")
        w_function_original.to_pickle(intermediate_dir + "w_function_original.pkl")
        apex_to_OT.to_pickle(intermediate_dir + "apex_to_OT.pkl")
        apex_to_base.to_pickle(intermediate_dir + "apex_to_base.pkl")

        if cfg.blend_at_septum:
            septal_coord.to_pickle(intermediate_dir + "septal_coord.pkl")

        if cfg.calculate_sheet or cfg.calculate_transverse:
            beta.to_pickle(intermediate_dir + "beta.pkl")

    if cfg.prepare_h5:
        logger.info("Saving h5 file for use in LBM pipeline")
        full_array = full.reconstruct(numpy=True)
        with h5py.File(results_dir + "/" + main_name + ".h5", "w") as h5f:
            h5f.create_dataset("grid", data=full_array, dtype='i4')
            h5f.create_dataset("fibers", data=fiber.vectors.cpu().numpy(), dtype='f4')


    logger.info(f"Pipeline runtime: {time.perf_counter() - start:.2f} seconds")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception:
        logger.exception(f"Unhandled Exception occurred.")
        raise