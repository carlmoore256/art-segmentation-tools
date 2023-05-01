from segment_anything import SamAutomaticMaskGenerator
from image import Image, alpha_blend_images
from mask import AnnotationMask, Mask
import numpy as np

class SegmentedImage():

  def __init__(self, image: Image):
    self.image = image
    self.masks = None

  def segment(self, mask_generator : SamAutomaticMaskGenerator):
    input_data = self.image.get_writeable_data()
    raw_masks = mask_generator.generate(input_data)
    if len(raw_masks) == 0:
      print(f'[!] Error computing masks, none found!')
      return
    self.masks = [AnnotationMask(m) for m in raw_masks]

  def visualize_masks(self, opacity=0.8):
    if self.masks is None:
      raise Exception("[!] Masks have't been computed yet, run segment()")
    sorted_masks = sorted(self.masks, key=(lambda x: x.area), reverse=True)
    new_image = self.image.copy()
    for mask in sorted_masks:
      new_image = alpha_blend_images(new_image, mask.to_image(4, color=None), opacity)
    return new_image

  def get_masked(self, index):
    if self.masks is None:
      raise Exception("[!] Masks have't been computed yet, run compute_masks()")
    if index < 0 or index > len(self.masks) - 1:
      raise Exception(f"[!] Index {index} out of range")
    return self.image.new_from_mask(self.masks[index])

  def unmasked_area_mask(self):
    unmasked = np.ones((self.image.shape[0], self.image.shape[1]))
    for mask in self.masks:
      unmasked[mask.mask_data] = 0
    return Mask(unmasked)

  def get_background(self):
    unmasked = self.unmasked_area_mask()
    background = self.image.new_from_mask(unmasked)
    return background

  def __len__(self):
    if self.masks is None:
      raise Exception("[!] Masks have't been computed yet, run compute_masks()")
    return len(self.masks)

  def __repr__(self):
    if self.masks is None:
      return f"SegmentedImage [not yet segmented, run .segment()]"
    return f"SegmentedImage | {len(self.masks)} masks"

  def __getitem__(self, index):
    if self.masks is None:
      raise Exception("[!] Masks have't been computed yet, run compute_masks()")
    if index < 0 or index > len(self.masks) - 1:
      raise Exception(f"[!] Index {index} out of range")
    return self.masks[index]