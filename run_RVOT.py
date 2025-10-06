import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import pickle 
import os
import SimpleITK as sitk
import numpy as np
from scipy.spatial import KDTree
import torch
from Sparse import SparseVolume, SparseField, SparseVectors
import h5py
from uvc import *
from fibers import *


logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path= "configs", config_name="univentricular_config")
def run_pipeline_RVOT(cfg: DictConfig) -> None:

    start = time.perf_counter()
    logger.info("================ Starting Pipeline For Right Ventricular Fibers With Outflow Tract (Doste, 2019) ================")
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

    logger.info("Checking that all points in volume are 6-connected")
    full = full.eliminate_non_connected_points()
    full.to_pickle(results_dir + "/full_volume.pkl")

    logger.info("Extracting and saving new border")
    new_border = full.extract_edge_points()
    new_border = initial_border.values_for_closest_points(new_border)
    new_border.to_pickle(results_dir + "/new_border.pkl")

    # different boundary conditions
    endo = new_border.extract_isosurface(cfg.labels.RV_endo_label)
    epi = new_border.extract_isosurface(cfg.labels.epi_label)
    base = new_border.extract_isosurface(cfg.labels.base_label)

    logger.info("Defining complete apex by linking the apex landmark on epicardium to the endocardium")
    if cfg.files.apex_path is None:
        logger.info("Apex path is not provided. Switching to automatic apex detection.")
        initial_apex = find_initial_apex_univentricular(full, base, epi)
    else:
        initial_apex = epi.closest_indices_from_fcsv(cfg.files.apex_path)
    complete_apex = get_new_apex(full, initial_apex, endo)
    complete_apex.to_pickle(results_dir + "/complete_apex.pkl")

    logger.info("Extracting intervalvular bridge")
    pulmonary, tricuspid = base.extract_closest_connected_surfaces([cfg.files.pulmonary_path, cfg.files.tricuspid_path])
    bridge_path= extract_bridge_path(full, pulmonary, tricuspid, rotation_angle=cfg.bridge_rotation_angle)
    intervalvular_coord = calculate_eikonal_field(full, border0=bridge_path, border1=complete_apex)
    mask = intervalvular_coord.values < cfg.intervalvular_threshold
    bridge = full.extract_points_based_on_mask(mask)
    univalve = full.extract_points_based_on_mask(~mask)
    bridge,univalve = exchange_non_connected_points(bridge, univalve)


    logger.info("Defining intervalvular and univalve edges")
    pulmonary_bridge_edge, tricuspid_bridge_edge, close_to_septum_edge, far_from_septum_edge = define_intervalvular_edges(bridge, pulmonary, tricuspid, endo, epi, complete_apex)
    univalve_edge = univalve.extract_edge_points()
    new_points_on_univalve_edge_mask = ~(new_border.check_point_membership(univalve_edge.points))
    new_univalve_edge_labels = new_border.values_for_closest_points(univalve_edge).values
    new_univalve_edge_labels[new_points_on_univalve_edge_mask] = cfg.labels.base_label
    univalve_edge = SparseField.from_Sparse_with_values(univalve_edge, new_univalve_edge_labels)
    univalve_base = univalve_edge.extract_isosurface(cfg.labels.base_label)


    logger.info("Calculating apicobasal coordinates on the intervalvular bridge")
    apicobasal_bridge = calculate_eikonal_field(bridge, border0=far_from_septum_edge, border1=close_to_septum_edge)
    apicobasal_bridge.values = 0.25 * apicobasal_bridge.values + 1  #range [1, 1.25]
    logger.info("Calulating apicobasal coordinates on the univalve ventricle (without the bridge)")
    apicobasal_univalve = calculate_eikonal_field(univalve, border0=complete_apex, border1=univalve_base)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        # smooth seperately to avoid smoothing across the discontinuity where voxels with values 1.25 meet voxels with value 1.
        apicobasal_bridge = apicobasal_bridge.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[1, 1.25])
        apicobasal_univalve = apicobasal_univalve.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
    logger.info("Joining apicobasal coordinates")
    apicobasal = SparseField.combine([apicobasal_bridge, apicobasal_univalve])
    apicobasal = apicobasal.reorder(full)

    #transmural
    logger.info("Calculating transmural coordinates")
    transmural = calculate_eikonal_field(full, border0=endo, border1=epi)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        transmural = transmural.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])

    #rotational
    logger.info("Calculating rotational coordinates on the univalve ventricle (without the bridge)")
    rotational_reference = full.closest_indices_from_fcsv(cfg.files.rotational_reference_path)
    iter_smooth_rot = cfg.smooth.iter if cfg.smooth.use else 0
    rotational_univalve = calculate_rotational_advanced(univalve, univalve_edge, initial_apex[0], rotational_reference[0], univalve_edge.get_centroid(), smooth_iter=iter_smooth_rot)
    logger.info("Calculating rotational coordinates on the intervalvular bridge")
    rotational_bridge = calculate_eikonal_field(bridge, border0 = pulmonary_bridge_edge, border1=tricuspid_bridge_edge)
    rotational_bridge_new_values = rotational_bridge.values * 0.25 + 1  # range [1, 1.25]
    rotational_bridge = SparseField.from_Sparse_with_values(bridge, rotational_bridge_new_values)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        rotational_bridge = rotational_bridge.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[1, 1.25])
    logger.info("Joining apicobasal coordinates")
    rotational = SparseField.combine([rotational_univalve, rotational_bridge])
    rotational = rotational.reorder(full)

    apicobasal.to_pickle(uvc_dir + "apicobasal.pkl")
    transmural.to_pickle(uvc_dir + "transmural.pkl")
    rotational.to_pickle(uvc_dir + "rotational.pkl")

    if cfg.debug:
        bridge_path.to_pickle(intermediate_dir + "bridge_path.pkl")
        bridge.to_pickle(intermediate_dir + "bridge.pkl")
        pulmonary.to_pickle(intermediate_dir + "pulmonary.pkl")
        tricuspid.to_pickle(intermediate_dir + "tricuspid.pkl")
        univalve.to_pickle(intermediate_dir + "univalve.pkl")
        apicobasal_univalve.to_pickle(intermediate_dir + "apicobasal_univalve.pkl")
        apicobasal_bridge.to_pickle(intermediate_dir + "apicobasal_bridge.pkl")

    if not cfg.calculate_fiber:
        logger.info("Skipping fiber calculation as per configuration")
        logger.info(f"Pipeline runtime: {time.perf_counter() - start:.2f} seconds")
        return

    logger.info("calculating apex_to_OT, apex_to_base and w_function")
    apex_to_OT = calculate_eikonal_field(full, border0=pulmonary, border1=complete_apex)
    apex_to_base = calculate_eikonal_field(full, border0=tricuspid, border1=complete_apex)
    w_function_original = calculate_eikonal_field(full, border0=pulmonary, border1=SparseVolume.combine([tricuspid, complete_apex]))
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

    apicobasal_vectors = w_values[..., None] * basal_vectors + (1 - w_values[..., None]) * OT_vectors
    norms = torch.linalg.norm(apicobasal_vectors, dim=1)
    norms[norms == 0] = 1
    apicobasal_vectors = apicobasal_vectors / norms[..., None]
    apicobasal_vectors = SparseVectors.from_Sparse_with_vectors(basal_vectors_sparse, apicobasal_vectors)

    logger.info("Calculating transmural vectors")
    #to respect Doste method, transmural vectors go into RV endocardium
    transmural_values_to_derive = 1 - transmural.values
    transmural_to_derive = SparseField.from_Sparse_with_values(transmural, transmural_values_to_derive)
    transmural_vectors = transmural_to_derive.gradient()

    #fiber assignment
    logger.info("Calculating local system")
    e_l, e_t, e_c = get_local_system(apicobasal_vectors, transmural_vectors)

    logger.info("Calculating alpha and beta values")
    beta_values = None
    alpha_values = get_alpha_values_doste(transmural, w_function, alpha_endo=cfg.angles.alpha_RV_endo, alpha_epi=cfg.angles.alpha_RV_epi, alpha_OT_endo=cfg.angles.alpha_OT_endo, alpha_OT_epi=cfg.angles.alpha_OT_epi)
    if cfg.calculate_transverse or cfg.calculate_sheet:
        beta_values = get_beta_values_doste(transmural, w_function, beta_endo=cfg.angles.beta_RV_endo, beta_epi=cfg.angles.beta_RV_epi, beta_OT_endo=cfg.angles.beta_OT_endo, beta_OT_epi=cfg.angles.beta_OT_epi)
    
    logger.info("Calculating fiber, sheet and transverse vectors")
    fiber, sheet, transverse = calculate_fibers(e_c, e_l, e_t, alpha_values, beta_values, calculate_sheet=cfg.calculate_sheet, calculate_transverse=cfg.calculate_transverse)

    logger.info("Interpolating missing fibers")
    fiber.interpolate_missing_vectors()

    logger.info("Saving final results to pickle files")
    fiber.to_pickle(results_dir + "/fiber.pkl")
    if cfg.calculate_sheet:
        sheet.interpolate_missing_vectors()
        sheet.to_pickle(results_dir + "/sheet.pkl")
    if cfg.calculate_transverse:
        transverse.interpolate_missing_vectors()
        transverse.to_pickle(results_dir + "/transverse.pkl")

    if cfg.debug:
        e_l.to_pickle(intermediate_dir + "e_l.pkl")
        e_t.to_pickle(intermediate_dir + "e_t.pkl")
        e_c.to_pickle(intermediate_dir + "e_c.pkl")
        apex_to_base.to_pickle(intermediate_dir + "apex_to_base.pkl")
        apex_to_OT.to_pickle(intermediate_dir + "apex_to_OT.pkl")
        alpha_values.to_pickle(intermediate_dir + "alpha_values.pkl")
        w_function_original.to_pickle(intermediate_dir + "w_function_original.pkl")
        w_function.to_pickle(intermediate_dir + "w_function.pkl")
        if cfg.calculate_transverse or cfg.calculate_sheet:
            beta_values.to_pickle(intermediate_dir + "beta_values.pkl")

    if cfg.prepare_h5:
        logger.info("Saving h5 file for use in LBM pipeline")
        full_array = full.reconstruct(numpy=True)
        with h5py.File(results_dir + "/" + main_name + ".h5", "w") as h5f:
            h5f.create_dataset("grid", data=full_array, dtype='i4')
            h5f.create_dataset("fibers", data=fiber.vectors.cpu().numpy(), dtype='f4')

    logger.info(f"Pipeline runtime: {time.perf_counter() - start:.2f} seconds")

if __name__ == "__main__":
    try:
        run_pipeline_RVOT()
    except Exception:
        logger.exception(f"Unhandled Exception occurred.")
        raise