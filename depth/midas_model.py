from depth.mde_model import MDEModel
import torch
from image_processing import convert_image_float32, convert_image_uint8
import numpy as np

class MidasModel(MDEModel):

    def __init__(self, model_name: str, device: str):
        if device.startswith("mps"):
            print(f'[!] Device {device} not supported for MidasModel, using cpu instead')
            device = "cpu"
        super().__init__(model_name, device)
    
    @property
    def valid_models(self):
        return ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]

    def load_transform(self):
        transform = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_name == "DPT_Large" or self.model_name == "DPT_Hybrid":
            return transform.dpt_transform
        else:
            return transform.small_transform

    def load_model(self):
        model = torch.hub.load("intel-isl/MiDaS", self.model_name)
        model.to(self.device)
        model.eval()
        return model

    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        image = convert_image_uint8(image)
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
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
    
    def predict_batch_depth(self, images: np.ndarray) -> np.ndarray:
        images = convert_image_uint8(images)
        input_batch = self.transform(images).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=images.shape[1:3],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        output = (output - np.min(output)) / (np.max(output) - np.min(output))
        output = np.expand_dims(output, -1)
        return output