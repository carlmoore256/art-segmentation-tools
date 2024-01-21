# append the path of art segmentation tools

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from bs4 import BeautifulSoup
from tqdm import tqdm
import io
from joblib import Parallel, delayed
import argparse
import os

from segmentation import load_model
from image import Image, alpha_blend_images
from mask import Mask, AnnotationMask
from segmented_image import SegmentedImage
from segment_anything import SamAutomaticMaskGenerator

SEGMENTATION_MODEL_PATH = "./art-segmentation-tools/data/sam_vit_h_4b8939.pth"


# "DPT_Large": MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# "DPT_Hybrid": MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# "MiDaS_small": MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
MIDAS_TYPE = "DPT_Large"
MIDAS = None
TRANSFORM = None

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def predict_depthmap(image : np.ndarray):
    global MIDAS, TRANSFORM, MIDAS_TYPE
    input_batch = TRANSFORM(image).to(device)
    with torch.no_grad():
        prediction = MIDAS(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    output = (output - np.min(output)) / (np.max(output) - np.min(output))
    output = np.expand_dims(output, -1)
    return output


def create_depth_map(image: np.ndarray, model_type="DPT_Large"):
    assert image.dtype == np.uint8
    assert model_type in ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    global MIDAS, TRANSFORM, MIDAS_TYPE
    if MIDAS is None or MIDAS_TYPE != model_type or TRANSFORM is None:
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        TRANSFORM = transform
        MIDAS = midas
    depthmap = predict_depthmap(image)
    return depthmap

def create_depth_map_from_file(image_path, model_type="DPT_Large"):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return create_depth_map(image, model_type)


def create_segmented_image(image_path):
    SEGMENTATION_MODEL = load_model(checkpoint=SEGMENTATION_MODEL_PATH)
    mask_generator = SamAutomaticMaskGenerator(
        model=SEGMENTATION_MODEL,
        # (32) Number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.
        points_per_side=32,
        # (64) Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory
        points_per_batch=32,
        # (0.88) A filtering threshold in [0,1], using the model's predicted mask quality
        pred_iou_thresh=0.88,
        # (0.95) A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
        stability_score_thresh=0.95,
        # (1.0) The amount to shift the cutoff when calculated the stability score
        stability_score_offset=1.0,
        # (0.7) The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
        box_nms_thresh=0.3,
        # (1) If > 0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
        crop_n_layers=0,
        # (0.7) The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
        crop_nms_thresh=0.7,
        # (512/1500, ~0.3413) Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
        crop_overlap_ratio=512 / 1500,
        # (1) The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
        crop_n_points_downscale_factor=1,
        # (None) A list over explicit grids of points used for sampling, normalized to [0,1]. The nth grid in the list is used in the nth crop layer. Exclusive with points_per_side
        point_grids=None,
        # (0) If > 0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
        min_mask_region_area=100,
        # ("binary_mask")  The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For large resolutions, 'binary_mask' may consume large amounts of memory.
        output_mode="binary_mask"
    )
    image = Image(image_path)
    print(f"Loaded image with dimensions: {image.shape}")
    seg_image = SegmentedImage(image)
    seg_image.segment(mask_generator)
    print(f'Generated segmented image: {seg_image}')
    return seg_image


def refine_depthmap(segments_image_normalized, depthmap, blend=0.5):
    output_depth = depthmap[:, :].copy()
    distinct_gradations = np.unique(segments_image_normalized)
    for gradation in distinct_gradations:
        # within each gradation, average the depth value
        gradation_mask = segments_image_normalized[:, :, 0] == gradation
        gradation_mask = np.expand_dims(gradation_mask, -1)
        # gradation_mask = ~gradation_mask
        gradation_depth = depthmap * gradation_mask
        gradation_depth = cv2.erode(
            gradation_depth, np.ones((3, 3), np.uint8), iterations=3)
        nonzero_depth = gradation_depth[gradation_depth > 0]
        avg_depth = np.mean(nonzero_depth)
        output_depth[gradation_mask] = avg_depth * blend + \
            output_depth[gradation_mask] * (1 - blend)
    return output_depth


def encode_image_segments(seg_image):
    # Initialize the image array
    segments_image = np.zeros(
        (seg_image.image.width, seg_image.image.height, 3), dtype=np.uint8)
    segments_image_normalized = np.zeros(
        (seg_image.image.width, seg_image.image.height, 1), dtype=np.float32)

    def get_next_color(current_color):
        r, g, b = current_color
        if b < 255:
            b += 1
        elif g < 255:
            g += 1
            b = 0
        elif r < 255:
            r += 1
            g = 0
            b = 0
        return r, g, b
    current_color = (0, 0, 0)
    # segments_image[:, :, :] = current_color

    for i, mask in enumerate(seg_image.masks):
        mask_arr = mask.to_array()
        mask_arr = mask_arr[:, :, 0]
        y_indices, x_indices = np.where(mask_arr > 0)
        for y, x in zip(y_indices, x_indices):
            segments_image[y, x, :] = current_color
            segments_image_normalized[y, x, :] = i / len(seg_image.masks)
        current_color = get_next_color(current_color)
    return segments_image, segments_image_normalized


def resize_image(image, max_px=2000):
    if image.shape[0] > image.shape[1]:
        ratio = image.shape[0] / image.shape[1]
        new_height = max_px
        new_width = int(new_height / ratio)
    else:
        ratio = image.shape[1] / image.shape[0]
        new_width = max_px
        new_height = int(new_width / ratio)
    return cv2.resize(image, (new_width, new_height))


def save_output(refined_depthmap, segments_image, image_path, outpath, filename, max_px=2000):
    print("Saving output...")
    color_path = os.path.join(outpath, f'{filename}.color.png')
    depth_path = os.path.join(outpath, f'{filename}.depth.png')
    segments_path = os.path.join(outpath, f'{filename}.segments.png')
    refined_depthmap = (refined_depthmap * 255).astype('uint8')
    depthmap = cv2.cvtColor(refined_depthmap, cv2.COLOR_GRAY2RGB)
    if max_px is not None:
        depthmap = resize_image(depthmap, max_px)
    cv2.imwrite(depth_path, depthmap)

    color_image = cv2.imread(image_path)
    if max_px is not None:
        color_image = resize_image(color_image, max_px)
    cv2.imwrite(color_path, color_image)

    if max_px is not None:
        segments_image = resize_image(segments_image, max_px)
    cv2.imwrite(segments_path, segments_image)
    print(f'Saved color image to {color_path} and depth image to {depth_path}')


def pipeline(image_path, options):
    seg_image = create_segmented_image(image_path)
    segments_image, segments_image_normalized = encode_image_segments(
        seg_image)
    depthmap = create_depth_map_from_file(image_path)
    refined_depthmap = refine_depthmap(
        segments_image_normalized, depthmap, options['blend'])
    save_output(refined_depthmap, segments_image, image_path,
                options['outpath'], options['filename'], options['max_px'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Create depth map from image')
    parser.add_argument('--image', type=str,
                        help='Path to image file')
    parser.add_argument('--outpath', type=str,
                        help='Path to output folder')
    parser.add_argument('--filename', type=str,
                        help='Output filename')
    parser.add_argument('--blend', type=float,
                        help='Blend factor')
    parser.add_argument('--max_px', type=int,
                        help='Max dimension of image')
    args = parser.parse_args()

    options = {
        'image': args.image,
        'outpath': args.outpath,
        'filename': args.filename,
        'blend': args.blend,
        'max_px': args.max_px
    }

    pipeline(args.image, options)
