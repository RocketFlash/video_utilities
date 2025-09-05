import cv2
import numpy as np
from typing import Dict, Any
from .base import FramePreprocessor

class GaussianBlurPreprocessor(FramePreprocessor):
    """Apply Gaussian blur."""
    
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
            
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), self.sigma)
    
    def get_name(self) -> str:
        return "gaussian_blur"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'GaussianBlurPreprocessor',
            'kernel_size': self.kernel_size,
            'sigma': self.sigma
        }

class MedianBlurPreprocessor(FramePreprocessor):
    """Apply median blur."""
    
    def __init__(self, kernel_size: int = 5):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")
        self.kernel_size = kernel_size
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(frame, self.kernel_size)
    
    def get_name(self) -> str:
        return "median_blur"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'MedianBlurPreprocessor',
            'kernel_size': self.kernel_size
        }

class BilateralFilterPreprocessor(FramePreprocessor):
    """Apply bilateral filter for noise reduction while preserving edges."""
    
    def __init__(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(frame, self.d, self.sigma_color, self.sigma_space)
    
    def get_name(self) -> str:
        return "bilateral_filter"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'BilateralFilterPreprocessor',
            'd': self.d,
            'sigma_color': self.sigma_color,
            'sigma_space': self.sigma_space
        }