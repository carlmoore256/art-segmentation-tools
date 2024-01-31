# easy script to convert an image into a segmented svg
from segmentation import load_model
from image import Image, alpha_blend_images
from mask import Mask, AnnotationMask
from segmented_image import SegmentedImage
from segment_anything import SamAutomaticMaskGenerator
from product import export_bundle
import argparse
import json

SEGMENTATION_MODEL = load_model()

# mask_generator = SamAutomaticMaskGenerator(
#     model=SEGMENTATION_MODEL,
#     # (32) Number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.
#     points_per_side=32,
#     # (64) Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory
#     points_per_batch=32,
#     # (0.88) A filtering threshold in [0,1], using the model's predicted mask quality
#     pred_iou_thresh=0.88,
#     # (0.95) A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
#     stability_score_thresh=0.95,
#     # (1.0) The amount to shift the cutoff when calculated the stability score
#     stability_score_offset=1.0,
#     # (0.7) The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
#     box_nms_thresh=0.3,
#     # (1) If > 0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
#     crop_n_layers=0,
#     # (0.7) The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
#     crop_nms_thresh=0.7,
#     # (512/1500, ~0.3413) Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
#     crop_overlap_ratio=512 / 1500,
#     # (1) The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
#     crop_n_points_downscale_factor=1,
#     # (None) A list over explicit grids of points used for sampling, normalized to [0,1]. The nth grid in the list is used in the nth crop layer. Exclusive with points_per_side
#     point_grids=None,
#     # (0) If > 0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
#     min_mask_region_area=100,
#     # ("binary_mask")  The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For large resolutions, 'binary_mask' may consume large amounts of memory.
#     output_mode="binary_mask"
# )

def configure_sam_parameters(detail_level, segment_stability, complexity_handling):
    """
    Configure parameters for the Segment Anything Model (SAM) to segment abstract artworks.

    Parameters:
    - detail_level (int): Determines the granularity of the segmentation. 
      Range: 1 to 10, where 1 is the least detailed (coarser segmentation) 
      and 10 is the most detailed (finer segmentation).
      
    - segment_stability (int): Controls the stability and consistency of the segments. 
      This parameter influences the 'pred_iou_thresh' and 'stability_score_thresh' of the model.
      Range: 1 to 10, where 1 results in less stable and consistent segments, 
      and 10 ensures maximum stability and consistency in segmentation. The values are
      scaled using a power function to fit within the 0-1 range for 'stability_score_thresh'.
      
    - complexity_handling (int): Adjusts how the model handles complex and overlapping patterns in abstract art.
      Range: 1 to 10, where 1 indicates minimal handling of complexity (suitable for simpler artworks), 
      and 10 indicates advanced handling of complex, overlapping patterns.

    Returns:
    - dict: A dictionary of parameters configured for SAM, tailored for segmenting abstract artworks.

    Example usage:
    sam_params = configure_sam_parameters(detail_level=7, segment_stability=6, complexity_handling=8)
    """

    # Mapping detail_level to points_per_side
    points_per_side = 16 if detail_level < 5 else 32 if detail_level < 8 else 64

    # Mapping segment_stability to pred_iou_thresh and stability_score_thresh
    # Using a power function to ensure values are within 0-1
    base_threshold = 0.3 ** (1 / (segment_stability + 1))
    pred_iou_thresh = base_threshold
    stability_score_thresh = min(1.0, base_threshold + 0.05)

    # Mapping complexity_handling to box_nms_thresh and min_mask_region_area
    box_nms_thresh = 0.7 if complexity_handling < 5 else 0.5 if complexity_handling < 8 else 0.3
    # min_mask_region_area = 50 if complexity_handling < 5 else 100 if complexity_handling < 8 else 200
    min_mask_region_area = 100
    
    # Setting other parameters with default or derived values
    points_per_batch = 32
    stability_score_offset = 1.0
    crop_n_layers = 0
    crop_nms_thresh = 0.7
    crop_overlap_ratio = 512 / 1500
    crop_n_points_downscale_factor = 1
    point_grids = None
    output_mode = "binary_mask"

    return {
        "points_per_side": points_per_side,
        "points_per_batch": points_per_batch,
        "pred_iou_thresh": pred_iou_thresh,
        "stability_score_thresh": stability_score_thresh,
        "stability_score_offset": stability_score_offset,
        "box_nms_thresh": box_nms_thresh,
        "crop_n_layers": crop_n_layers,
        "crop_nms_thresh": crop_nms_thresh,
        "crop_overlap_ratio": crop_overlap_ratio,
        "crop_n_points_downscale_factor": crop_n_points_downscale_factor,
        "point_grids": point_grids,
        "min_mask_region_area": min_mask_region_area,
        "output_mode": output_mode
    }



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--outdir", type=str, required=True)
    args.add_argument("--background", type=str, required=False)
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("--resolution", type=int, required=False, default=2500)
    args.add_argument("--detail_level", type=int, required=False, default=7)
    args.add_argument("--segment_stability", type=int, required=False, default=6)
    args.add_argument("--complexity_handling", type=int, required=False, default=8)
    args = args.parse_args()
    image = Image(args.input)
    image.pad_to_square()
    image.resize((args.resolution, args.resolution))
    background = Image(args.background) if args.background else None
    seg_image = SegmentedImage(image)

    parameters = configure_sam_parameters(args.detail_level, args.segment_stability, args.complexity_handling)
    print(f'Configured mask generator with parameters: {json.dumps(parameters, indent=2)}')
    mask_generator = SamAutomaticMaskGenerator(model=SEGMENTATION_MODEL, **parameters)
    if background:
        seg_image.set_background(background)
    seg_image.segment(mask_generator)
    export_bundle(seg_image, args.name, args.outdir, overwrite=args.overwrite)
    print(f'Exported to {args.outdir}')
