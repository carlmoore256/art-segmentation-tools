from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class MDEModel(ABC):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        if self.verify_model(model_name):
            self.model = self.load_model()
            self.transform = self.load_transform()

    @property
    @abstractmethod
    def valid_models(self):
        pass

    def verify_model(self, model_name: str):
        if model_name not in self.valid_models:
            raise ValueError(f"Invalid model name, use {self.valid_models}")
        return True

    @abstractmethod
    def load_transform(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_batch_depth(self, images: np.ndarray) -> np.ndarray:
        pass
