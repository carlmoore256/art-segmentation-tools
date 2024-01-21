import torch

# TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# TORCH_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
TORCH_DEVICE = 'cpu'
print(f"Using device: {TORCH_DEVICE}")

SEGMENTATION_MODEL_PATH = "data/sam_vit_h_4b8939.pth"

DEFAULT_MASK_GENERATOR_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,
}

