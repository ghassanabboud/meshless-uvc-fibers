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
    
    full_septal_surface = SparseVolume.combine([lv_septal_surface, rv_septal_surface])

    logger.info("Linking the apex to the endocardium on each ventricle")
    lv_complete_apex = get_new_apex(lv, lv_initial_apex, lv_endo)
    rv_complete_apex = get_new_apex(rv, rv_initial_apex, rv_endo)

    #step 4: apicobasal coordinates
    logger.info("Calculating apicobasal coordinates for RV and LV")
    apicobasal_rv = calculate_eikonal_field(rv, border0=rv_complete_apex, border1=rv_base)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        apicobasal_rv = apicobasal_rv.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
    apicobasal_lv = calculate_eikonal_field(lv, border0=lv_complete_apex, border1=lv_base)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        apicobasal_lv = apicobasal_lv.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
    apicobasal = SparseField.combine([apicobasal_lv, apicobasal_rv])
    apicobasal = apicobasal.reorder(full)

    #step 5: calculating ridge coordinates and separating volumes into free wall and septum for RV and LV
    logger.info("Calculating ridge coordinates and separating volumes into free wall and septum for RV and LV")
    ridge_coord = calculate_eikonal_field(full, border0=epi, border1=full_septal_surface)

    rv_free_volume = full.extract_points_based_on_mask((transventricular.values ==1) & (ridge_coord.values <= 0.5))
    rv_septum_volume = full.extract_points_based_on_mask((transventricular.values ==1) & (ridge_coord.values > 0.5))
    rv_free_volume, rv_septum_volume = exchange_non_connected_points(rv_free_volume, rv_septum_volume)

    lv_free_volume = full.extract_points_based_on_mask((transventricular.values ==0) & (ridge_coord.values <= 0.5))
    lv_septum_volume = full.extract_points_based_on_mask((transventricular.values ==0) & (ridge_coord.values > 0.5))
    lv_free_volume, lv_septum_volume = exchange_non_connected_points(lv_free_volume, lv_septum_volume)

    #step 6: rotational coordinates
    logger.info("Calculating rotational coordinates for RV and LV")
    iter_smooth_rot = cfg.smooth.iter if cfg.smooth.use else 0
    lambda_param = cfg.smooth.lambda_param if cfg.smooth.use else 0.0
    rotational_rv = calculate_rotational_cobiveco( rv_free_volume, rv_septum_volume, rv_border, rv_initial_apex[0], rv_center, full_base_center, iter_smooth_rot, lambda_param, reverse=True)
    rotational_lv = calculate_rotational_cobiveco( lv_free_volume, lv_septum_volume, lv_border, lv_initial_apex[0], lv_center, full_base_center, iter_smooth_rot, lambda_param)
    rotational = SparseField.combine([rotational_lv, rotational_rv])
    rotational = rotational.reorder(full)

    #step 7: transmural coordinates
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

    if not cfg.calculate_fiber:
        logger.info("Skipping fiber calculation as per configuration")
        logger.info(f"Pipeline runtime: {time.perf_counter() - start:.2f} seconds")
        return
    
    #step 8: gradient fields of coordinates
    logger.info("Calculating transmural vectors")
    # to match Doste, transmural vectors go out of LV endo and into RV endo.
    transmural_vectors_lv = transmural_lv.gradient()
    transmural_rv_to_derive_values = 1 - transmural_rv.values
    transmural_rv_to_derive = SparseField.from_Sparse_with_values(transmural_rv, transmural_rv_to_derive_values)
    transmural_vectors_rv = transmural_rv_to_derive.gradient()

    total_transmural_vectors = SparseVectors.combine([transmural_vectors_lv, transmural_vectors_rv])
    total_transmural_vectors = total_transmural_vectors.reorder(full)

    logger.info("Calculating apicobasal vectors for LV and RV")
    #to match Doste, apicobasal vectors go from base to apex.
    apicobasal_lv_to_derive_values = 1 - apicobasal_lv.values
    apicobasal_lv_to_derive = SparseField.from_Sparse_with_values(apicobasal_lv, apicobasal_lv_to_derive_values)
    apicobasal_vectors_lv = apicobasal_lv_to_derive.gradient()
    apicobasal_rv_to_derive_values = 1 - apicobasal_rv.values
    apicobasal_rv_to_derive = SparseField.from_Sparse_with_values(apicobasal_rv, apicobasal_rv_to_derive_values)
    apicobasal_vectors_rv = apicobasal_rv_to_derive.gradient()
    total_apicobasal_vectors = SparseVectors.combine([apicobasal_vectors_lv, apicobasal_vectors_rv])
    total_apicobasal_vectors = total_apicobasal_vectors.reorder(full)

    #step 9: alpha and beta values
    logger.info("Calculating alpha and beta values")

    alpha_lv = get_alpha_values(transmural_lv, alpha_endo=cfg.angles.alpha_LV_endo, alpha_epi=cfg.angles.alpha_LV_epi)
    alpha_rv = get_alpha_values(transmural_rv, alpha_endo=cfg.angles.alpha_RV_endo, alpha_epi=cfg.angles.alpha_RV_epi)
    alpha = SparseField.combine([alpha_lv, alpha_rv])
    alpha = alpha.reorder(full)
    beta = None
    
    if cfg.calculate_sheet or cfg.calculate_transverse:
        beta_lv = get_beta_values(transmural_lv, beta_endo=cfg.angles.beta_LV_endo, beta_epi=cfg.angles.beta_LV_epi)
        beta_rv = get_beta_values(transmural_rv, beta_endo=cfg.angles.beta_RV_endo, beta_epi=cfg.angles.beta_RV_epi)
        beta = SparseField.combine([beta_lv, beta_rv])
        beta = beta.reorder(full)


    #step 10: septal coordinates for blending alpha values in the septum (optional)
    if cfg.blend_at_septum:
        logger.info("Calculating septal coordinates for blending alpha values in the septum")
        septal_coord = calculate_eikonal_field(full, border0=SparseVolume.combine([rv_endo, lv_endo]), border1=full_septal_surface)
        if cfg.smooth.use:
            logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
            septal_coord = septal_coord.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])
        alpha_values = alpha.values * (1 - septal_coord.values) + cfg.angles.alpha_septal * septal_coord.values
        alpha = SparseField.from_Sparse_with_values(alpha, alpha_values)

        if cfg.calculate_sheet or cfg.calculate_transverse:
            beta_values = beta.values * (1 - septal_coord.values) + cfg.angles.beta_septal * septal_coord.values
            beta = SparseField.from_Sparse_with_values(beta, beta_values)

    #step 11: local coordinate system
    logger.info("Calculating local coordinate system")
    e_l, e_t, e_c = get_local_system(total_apicobasal_vectors, total_transmural_vectors)

    #step 12: fiber, sheet and transverse vectors through rotations
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