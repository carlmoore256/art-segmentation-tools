from bs4 import BeautifulSoup
from segment_anything import SamAutomaticMaskGenerator
from image import Image, alpha_blend_images
from mask import AnnotationMask, Mask
import numpy as np
from tqdm import tqdm


class SegmentedImage():

    def __init__(self, image: Image):
        self.image = image
        self.masks = None
        self.background = None

    def segment(self, mask_generator: SamAutomaticMaskGenerator):
        input_data = self.image.get_writeable_data()
        raw_masks = mask_generator.generate(input_data)
        if len(raw_masks) == 0:
            raise Exception(f'[!] Error computing masks, none found!')
        self.masks = [AnnotationMask(m) for m in raw_masks]

    def visualize_masks(self, opacity=0.8):
        if self.masks is None:
            raise Exception(
                "[!] Masks have't been computed yet, run segment()")
        sorted_masks = sorted(self.masks, key=(lambda x: x.area), reverse=True)
        new_image = self.image.copy()
        print(f'Generating mask visualization for {len(sorted_masks)} masks')
        for mask in tqdm(sorted_masks):
            new_image = alpha_blend_images(
                new_image, mask.to_image(4, color=None), opacity)
        return new_image

    def get_masked(self, index, with_alpha=True):
        if self.masks is None:
            raise Exception(
                "[!] Masks have't been computed yet, run compute_masks()")
        if index < 0 or index > len(self.masks) - 1:
            raise Exception(f"[!] Index {index} out of range")
        return self.image.new_from_mask(self.masks[index], with_alpha=with_alpha)

    def unmasked_area_mask(self):
        unmasked = np.ones((self.image.shape[0], self.image.shape[1]))
        for mask in self.masks:
            unmasked[mask.mask_data] = 0
        return Mask(unmasked)

    def get_background(self):
        if self.background is None:
            unmasked = self.unmasked_area_mask()
            # simple way to make a background by choosing the parts that haven't been segmented
            self.background = self.image.new_from_mask(unmasked)
        return self.background

    def set_background(self, background: Image):
        self.background = background

    def get_masks_by_area(self, descending=True):
        if self.masks is None:
            raise Exception(
                "[!] Masks have't been computed yet, run compute_masks()")
        return sorted(self.masks, key=(lambda x: x.area), reverse=descending)

    def __len__(self):
        if self.masks is None:
            raise Exception(
                "[!] Masks have't been computed yet, run compute_masks()")
        return len(self.masks)

    def __repr__(self):
        if self.masks is None:
            return f"SegmentedImage [not yet segmented, run .segment()]"
        return f"SegmentedImage | {len(self.masks)} masks"

    def __getitem__(self, index):
        if self.masks is None:
            raise Exception(
                "[!] Masks have't been computed yet, run compute_masks()")
        if index < 0 or index > len(self.masks) - 1:
            raise Exception(f"[!] Index {index} out of range")
        return self.masks[index]


def extract_image_from_soup(soup):
    image = soup.find("image")
    if image is None:
        raise ValueError("Could not find image in SVG")
    # convert base64 image to numpy array
    image = Image.from_base64(image["xlink:href"].split(",")[1])
    return image


def get_image_from_svg(svg_path):
    with open(svg_path, "r") as f:
        soup = BeautifulSoup(f.read(), "lxml")
        image_group = soup.find(attrs={"id": "image-orig"})
        if image_group is None:
            raise ValueError("Could not find image group in SVG")
        image = extract_image_from_soup(image_group)
        return image


# def load_segmented_image_from_svg(svg_path):
#     with open(svg_path, "r") as f:
#         soup = BeautifulSoup(f.read(), "xml")
#         image_group = soup.find(attrs={"id": "image-orig"})
#         if image_group is None:
#             raise ValueError("Could not find image group in SVG")

#         image = extract_image_from_soup(image_group)
#         image.show()
