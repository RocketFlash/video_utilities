import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .base import FramePreprocessor

class BrightnessContrastPreprocessor(FramePreprocessor):
    """Adjust brightness and contrast deterministically."""
    
    def __init__(self, brightness: float = 0, contrast: float = 1.0):
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
    
    def get_name(self) -> str:
        return "brightness_contrast"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'BrightnessContrastPreprocessor',
            'brightness': self.brightness,
            'contrast': self.contrast
        }

class CLAHEPreprocessor(FramePreprocessor):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        if clip_limit <= 0:
            raise ValueError("Clip limit must be positive")
        if any(x <= 0 for x in tile_grid_size):
            raise ValueError("Tile grid size must be positive")
            
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            # Convert to LAB, apply CLAHE to L channel, convert back
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            return self.clahe.apply(frame)
    
    def get_name(self) -> str:
        return "clahe"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'CLAHEPreprocessor',
            'clip_limit': self.clip_limit,
            'tile_grid_size': self.tile_grid_size
        }

class GammaPreprocessor(FramePreprocessor):
    """Apply gamma correction."""
    
    def __init__(self, gamma: float = 1.0):
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        self.gamma = gamma
        
        # Pre-compute lookup table for efficiency
        self.lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                           for i in range(256)]).astype(np.uint8)
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.LUT(frame, self.lut)
    
    def get_name(self) -> str:
        return "gamma"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'GammaPreprocessor',
            'gamma': self.gamma
        }

class HSVAdjustPreprocessor(FramePreprocessor):
    """Adjust HSV values."""
    
    def __init__(self, hue_shift: int = 0, saturation_scale: float = 1.0, value_scale: float = 1.0):
        self.hue_shift = hue_shift % 180  # Hue wraps around
        self.saturation_scale = max(0, saturation_scale)
        self.value_scale = max(0, value_scale)
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) != 3:
            raise ValueError("HSV adjustment requires color image")
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust hue
        hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift) % 180
        
        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_scale, 0, 255)
        
        # Adjust value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.value_scale, 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def get_name(self) -> str:
        return "hsv_adjust"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'HSVAdjustPreprocessor',
            'hue_shift': self.hue_shift,
            'saturation_scale': self.saturation_scale,
            'value_scale': self.value_scale
        }

class GrayscalePreprocessor(FramePreprocessor):
    """Convert to grayscale."""
    
    def __init__(self, keep_channels: bool = False):
        """
        Args:
            keep_channels: If True, return 3-channel grayscale, else single channel
        """
        self.keep_channels = keep_channels
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if self.keep_channels:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            return gray
        return frame
    
    def get_name(self) -> str:
        return "grayscale"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'GrayscalePreprocessor',
            'keep_channels': self.keep_channels
        }