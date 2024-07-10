import sys
import os

# sys.path.append()

script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(script_dir, "../Depth-Anything")
module_dir = os.path.abspath(module_dir)
sys.path.append(module_dir)


CACHE_DIR = os.path.join(module_dir, "torchhub")


from depth.mde_model import MDEModel
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything
from torchvision.transforms import Compose
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import contextlib


@contextlib.contextmanager
def temporary_directory_change(path):
    """Context manager to temporarily change the working directory."""
    original_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_path)


class DepthAnythingModel(MDEModel):
    @property
    def valid_models(self):
        return [
            "depth_anything_vits14",
            "depth_anything_vitb14",
            "depth_anything_vitl14",
        ]

    def load_transform(self):
        return Compose(
            [
                Resize(
                    width=518,  # 518 original
                    height=518,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def load_model(self):
        with temporary_directory_change(module_dir):
            model = (
                DepthAnything.from_pretrained(
                    f"LiheYoung/{self.model_name}", cache_dir=CACHE_DIR
                )
                .to(self.device)
                .eval()
            )
            return model

    def predict_depth(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        h, w = image.shape[:2]
        image = image / 255.0
        image = self.transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.model(image)
            # re-scale the image to the original size
            depth = F.interpolate(
                depth[None], (h, w), mode="bilinear", align_corners=False
            )[0, 0]

            if normalize:
                depth = depth / (255 * 3) # I dont even know the max
                # get the max depth
                d_min = depth.min()
                d_max = depth.max()
                depth = (depth - d_min) / (d_max - d_min)  # Normalize to 0-1
                depth = 1 - depth  # Invert
                depth = depth * (d_max - d_min) + d_min  # Rescale to original range
                if depth.max() < 1:
                    depth += 1e-5
                    
            depth = depth.cpu().numpy()
        depth = np.expand_dims(depth, -1)
        return depth

    def predict_batch_depth(self, images: np.ndarray) -> np.ndarray:
        h, w = images.shape[1:3]
        batch_size = images.shape[0]
        images = images / 255.0
        transformed_images = []
        for img in images:
            transformed_img = self.transform({'image': img})['image']
            transformed_images.append(torch.from_numpy(transformed_img))
        # Stack the transformed images into a batch
        images_batch = torch.stack(transformed_images).to(self.device)
        with torch.no_grad():
            depth = self.model(images_batch)
            # depth = depth.unsqueeze(1)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)
            depth = depth[0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # invert the depth map
            depth = 256 - depth
            depth = depth.cpu().numpy().astype(np.uint8)
        depth = np.expand_dims(depth, -1)
        return depth

    def __str__(self):
        return f"DepthAnythingModel({self.model_name}) on {self.device}"
