# easy script to convert an image into a segmented svg
from segmentation import load_model
from image import Image, alpha_blend_images
from mask import Mask, AnnotationMask
from segmented_image import SegmentedImage
from segment_anything import SamAutomaticMaskGenerator
from product import export_bundle
import argparse

SEGMENTATION_MODEL = load_model()

mask_generator = SamAutomaticMaskGenerator(
    model=SEGMENTATION_MODEL,
    points_per_side=32, # (32) Number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, 'point_grids' must provide explicit point sampling.
    points_per_batch=32, # (64) Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory
    pred_iou_thresh=0.88, # (0.88) A filtering threshold in [0,1], using the model's predicted mask quality
    stability_score_thresh=0.95, # (0.95) A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
    stability_score_offset=1.0, # (1.0) The amount to shift the cutoff when calculated the stability score
    box_nms_thresh=0.3, # (0.7) The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
    crop_n_layers=0, # (1) If > 0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
    crop_nms_thresh=0.7, # (0.7) The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
    crop_overlap_ratio= 512 / 1500, # (512/1500, ~0.3413) Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
    crop_n_points_downscale_factor=1, # (1) The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
    point_grids=None, # (None) A list over explicit grids of points used for sampling, normalized to [0,1]. The nth grid in the list is used in the nth crop layer. Exclusive with points_per_side
    min_mask_region_area=100, # (0) If > 0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.
    output_mode="binary_mask" # ("binary_mask")  The form masks are returned in. Can be 'binary_mask', 'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For large resolutions, 'binary_mask' may consume large amounts of memory.
)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--outdir", type=str, required=True)
    args.add_argument("--background", type=str, required=False)
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("--resolution", type=int, required=False, default=2500)
    args = args.parse_args()
    image = Image(args.input)
    image.pad_to_square()
    image.resize((args.resolution, args.resolution))
    background = Image(args.background) if args.background else None
    seg_image = SegmentedImage(image)
    if background:
        seg_image.set_background(background)
    seg_image.segment(mask_generator)
    export_bundle(seg_image, args.name, args.outdir, overwrite=args.overwrite)
    print(f'Exported to {args.outdir}')