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
def run_pipeline_LV(cfg: DictConfig) -> None:

    start = time.perf_counter()
    logger.info("================ Starting Pipeline For Left Ventricular Fibers ================")
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
    endo = new_border.extract_isosurface(cfg.labels.LV_endo_label)
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

    #apicobasal
    logger.info("Calculating apicobasal coordinates")
    apicobasal = calculate_eikonal_field(full, border0=complete_apex, border1=base)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        apicobasal = apicobasal.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])

    #transmural
    logger.info("Calculating transmural coordinates")
    transmural = calculate_eikonal_field(full, border0=endo, border1=epi)
    if cfg.smooth.use:
        logger.info(f"Applying forward smoothing with lambda={cfg.smooth.lambda_param} and iterations={cfg.smooth.iter}")
        transmural = transmural.forward_smooth(lambda_param=cfg.smooth.lambda_param, iter=cfg.smooth.iter, values_to_keep=[0, 1])

    #rotational
    logger.info("Calculating rotational coordinates")
    rotational_reference = full.closest_indices_from_fcsv(cfg.files.rotational_reference_path)
    iter_smooth_rot = cfg.smooth.iter if cfg.smooth.use else 0
    center = full.get_centroid()
    rotational = calculate_rotational_advanced(full, new_border, initial_apex[0], rotational_reference[0], center, smooth_iter=iter_smooth_rot, reverse=True)
    with open(results_dir + "/ventricle_center.pkl", "wb") as f:
        pickle.dump(center, f)

    apicobasal.to_pickle(uvc_dir + "apicobasal.pkl")
    rotational.to_pickle(uvc_dir + "rotational.pkl")
    transmural.to_pickle(uvc_dir + "transmural.pkl")

    if not cfg.calculate_fiber:
        logger.info("Skipping fiber calculation as per configuration")
        logger.info(f"Pipeline runtime: {time.perf_counter() - start:.2f} seconds")
        return

    logger.info("Calculating apicobasal vectors")
    #to respect Doste method, apicobasal go down from base to apex
    apicobasal_values_to_derive = 1- apicobasal.values
    apicobasal_to_derive = SparseField.from_Sparse_with_values(apicobasal, apicobasal_values_to_derive)
    apicobasal_vectors = apicobasal_to_derive.gradient()

    logger.info("Calculating transmural vectors")
    #to respect Doste method, transmural vectors go out of LV endocardium
    transmural_vectors = transmural.gradient()

    logger.info("Calculating local system")
    e_l, e_t, e_c = get_local_system(apicobasal_vectors, transmural_vectors)

    logger.info("Calculating alpha values")
    alpha_values = get_alpha_values(transmural, alpha_endo=cfg.angles.alpha_LV_endo, alpha_epi=cfg.angles.alpha_LV_epi)
    beta_values=None
    if cfg.calculate_transverse or cfg.calculate_sheet:
        logger.info("Calculating beta values")
        beta_values = get_beta_values(transmural, beta_endo=cfg.angles.beta_LV_endo, beta_epi=cfg.angles.beta_LV_epi)
    
    logger.info("Calculating fiber, sheet and transverse vectors")
    fiber, sheet, transverse = calculate_fibers(e_c, e_l, e_t, alpha_values, beta_values, calculate_sheet=cfg.calculate_sheet, calculate_transverse=cfg.calculate_transverse)
    fiber.interpolate_missing_vectors()
    logger.info("Interpolating missing fibers")

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
        alpha_values.to_pickle(intermediate_dir + "alpha_values.pkl")
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
        run_pipeline_LV()
    except Exception:
        logger.exception(f"Unhandled Exception occurred.")
        raise