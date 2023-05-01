import torch
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
from definitions import SEGMENTATION_MODEL_PATH

SEGMENTATION_MODEL = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(sam_model_type="vit_h"):
    global SEGMENTATION_MODEL
    SEGMENTATION_MODEL = sam_model_registry[sam_model_type](checkpoint=SEGMENTATION_MODEL_PATH)
    SEGMENTATION_MODEL.to(device=DEVICE)
    return SEGMENTATION_MODEL

def get_mask_generator(model=None, params=None):
    if model is None:
        model = SEGMENTATION_MODEL
    if params is None:
        return SamAutomaticMaskGenerator(model=model)
    return SamAutomaticMaskGenerator(model=model, **params)