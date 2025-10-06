from Sparse import *
import torch
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path= "configs", config_name="AHA_config")
def run_segments(cfg: DictConfig) -> None:

    apicobasal = SparseField.from_pickle(cfg.apicobasal_path)
    rotational = SparseField.from_pickle(cfg.rotational_path)

    mask_lv_segment_17 = apicobasal.values <= 0.2
    mask_lv_apical = (apicobasal.values > 0.2) & (apicobasal.values <= 0.4667)
    mask_lv_mid = (apicobasal.values > 0.4667) & (apicobasal.values <= 0.7333)
    mask_lv_basal = (apicobasal.values > 0.7333) & (apicobasal.values <= 1.0)

    mask_lv_segment_15 = mask_lv_apical & ((rotational.values <= (1/6 + 1/24)) | (rotational.values > (1-1/24)))
    mask_lv_segment_16 = mask_lv_apical & (rotational.values > (1/6 + 1/24)) & (rotational.values <= (1/6 + 1/24 + 1/4))
    mask_lv_segment_13 = mask_lv_apical & (rotational.values > (1/6 + 1/24 + 1/4)) & (rotational.values <= (1/6 + 1/24 + 2/4))
    mask_lv_segment_14 = mask_lv_apical & (rotational.values > (1/6 + 1/24 + 2/4)) & (rotational.values <= (1 - 1/24))

    mask_lv_segment_10 = mask_lv_mid & (rotational.values <= 1/6)
    mask_lv_segment_11 = mask_lv_mid & (rotational.values > 1/6) & (rotational.values <= 2/6)
    mask_lv_segment_12 = mask_lv_mid & (rotational.values > 2/6) & (rotational.values <= 3/6)
    mask_lv_segment_7 = mask_lv_mid & (rotational.values > 3/6) & (rotational.values <= 4/6)
    mask_lv_segment_8 = mask_lv_mid & (rotational.values > 4/6) & (rotational.values <= 5/6)
    mask_lv_segment_9 = mask_lv_mid & (rotational.values > 5/6) & (rotational.values <= 1.0)

    mask_lv_segment_4 = mask_lv_basal & (rotational.values <= 1/6)
    mask_lv_segment_5 = mask_lv_basal & (rotational.values > 1/6) & (rotational.values <= 2/6)
    mask_lv_segment_6 = mask_lv_basal & (rotational.values > 2/6) & (rotational.values <= 3/6)
    mask_lv_segment_1 = mask_lv_basal & (rotational.values >= 3/6) & (rotational.values <= 4/6)
    mask_lv_segment_2 = mask_lv_basal & (rotational.values > 4/6) & (rotational.values <= 5/6)
    mask_lv_segment_3 = mask_lv_basal & (rotational.values > 5/6) & (rotational.values <= 1.0)

    segment_values = torch.zeros_like(apicobasal.values, dtype=torch.float32)
    segment_values[mask_lv_segment_1] = 1
    segment_values[mask_lv_segment_2] = 2
    segment_values[mask_lv_segment_3] = 3
    segment_values[mask_lv_segment_4] = 4
    segment_values[mask_lv_segment_5] = 5
    segment_values[mask_lv_segment_6] = 6
    segment_values[mask_lv_segment_7] = 7
    segment_values[mask_lv_segment_8] = 8
    segment_values[mask_lv_segment_9] = 9
    segment_values[mask_lv_segment_10] = 10
    segment_values[mask_lv_segment_11] = 11
    segment_values[mask_lv_segment_12] = 12
    segment_values[mask_lv_segment_13] = 13
    segment_values[mask_lv_segment_14] = 14
    segment_values[mask_lv_segment_15] = 15
    segment_values[mask_lv_segment_16] = 16
    segment_values[mask_lv_segment_17] = 17

    aha_segments_lv = SparseField.from_Sparse_with_values(apicobasal, segment_values)
    aha_segments_lv.to_pickle(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/aha_segments.pkl")

if __name__ == "__main__":
    run_segments()