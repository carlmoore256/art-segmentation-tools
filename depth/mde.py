"""Monocular depth estimation (MDE)"""
import numpy as np
from enum import Enum
from depth.midas_model import MidasModel
from depth.depth_anything_model import DepthAnythingModel
from torch_utils import get_best_device

class MDEModelType(Enum):
    MIDAS_SMALL = "MiDaS_small"
    MIDAS_MEDIUM = "DPT_Hybrid"
    MIDAS_LARGE = "DPT_Large"
    DEPTH_ANYTHING_SMALL = "depth_anything_vits14"
    DEPTH_ANYTHING_MEDIUM = "depth_anything_vitb14"
    DEPTH_ANYTHING_LARGE = "depth_anything_vitl14"

# cache models
MIDAS_MODEL = None
DEPTH_ANYTHING_MODEL = None

# WARNING - if predicting depth map on MPS device, the image must be square

def predict_depth_midas(image: np.ndarray, model_name: str, device: str) -> np.ndarray:
    global MIDAS_MODEL
    if device.startswith("mps"):
        print(f'[!] Device {device} not supported for DepthAnythingModel, using cpu instead')
        device = "cpu"
    if MIDAS_MODEL is None:
        print(f'Creating new model {model_name}')
        MIDAS_MODEL = MidasModel(model_name, device)
    elif MIDAS_MODEL.model_name != model_name or MIDAS_MODEL.device != device:
        print(f'Creating new model {model_name}')
        MIDAS_MODEL = MidasModel(model_name, device)
    return MIDAS_MODEL.predict_depth(image)


def predict_depth_anything(image: np.ndarray, model_name: str, device: str) -> np.ndarray:
    global DEPTH_ANYTHING_MODEL
    if DEPTH_ANYTHING_MODEL is None:
        print(f'Creating new model {model_name}')
        DEPTH_ANYTHING_MODEL = DepthAnythingModel(model_name, device)
    elif DEPTH_ANYTHING_MODEL.model_name != model_name or DEPTH_ANYTHING_MODEL.device != device:
        print(f'Creating new model {model_name}')
        DEPTH_ANYTHING_MODEL = DepthAnythingModel(model_name, device)
    print(f'Predicting depth with model {DEPTH_ANYTHING_MODEL}')
    return DEPTH_ANYTHING_MODEL.predict_depth(image)


def predict_depth(
    image: np.ndarray,
    model: MDEModelType = MDEModelType.DEPTH_ANYTHING_LARGE,
    device: str = None,
) -> np.ndarray:
    """
    Predicts depth map from an image.

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: depth map
    """
    if device is None:
        device = str(get_best_device())
        print(f'Using device {device}')
    match model:
        case MDEModelType.MIDAS_SMALL:
            return predict_depth_midas(image, model.value, device)
        case MDEModelType.MIDAS_MEDIUM:
            return predict_depth_midas(image, model.value, device)
        case MDEModelType.MIDAS_LARGE:
            return predict_depth_midas(image, model.value, device)
        case MDEModelType.DEPTH_ANYTHING_SMALL:
            return predict_depth_anything(image, model.value, device)
        case MDEModelType.DEPTH_ANYTHING_MEDIUM:
            return predict_depth_anything(image, model.value, device)
        case MDEModelType.DEPTH_ANYTHING_LARGE:
            return predict_depth_anything(image, model.value, device)
        case _:
            raise ValueError("Invalid model name")
