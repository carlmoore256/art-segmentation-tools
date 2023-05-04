import sys
sys.path.append("inpaint-anything")

from stable_diffusion_inpaint import fill_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

from image import Image
from mask import Mask
from definitions import TORCH_DEVICE

def inpaint_image(image : Image, mask : Mask, text_prompt : str = None):
    img_filled = fill_img_with_sd(
        image.get_writeable_data(), 
        mask.get_writeable_data(), 
        text_prompt=text_prompt, 
        device=TORCH_DEVICE
    )
    return Image.from_data(img_filled)