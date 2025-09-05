import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .base import FramePreprocessor

class CropPreprocessor(FramePreprocessor):
    """Crop frame to specified region."""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        """
        Args:
            x1, y1, x2, y2: Crop coordinates in pixels
        """
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid crop coordinates: x1 < x2 and y1 < y2 required")
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Validate coordinates
        x1 = max(0, min(self.x1, w))
        y1 = max(0, min(self.y1, h))
        x2 = max(x1, min(self.x2, w))
        y2 = max(y1, min(self.y2, h))
        
        return frame[y1:y2, x1:x2]
    
    def get_name(self) -> str:
        return "crop"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'CropPreprocessor',
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2
        }

class ResizePreprocessor(FramePreprocessor):
    """Resize frame to target dimensions."""
    
    def __init__(self, width: int, height: int, interpolation=cv2.INTER_LINEAR):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        
        self.width = width
        self.height = height
        self.interpolation = interpolation
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self.width, self.height), interpolation=self.interpolation)
    
    def get_name(self) -> str:
        return "resize"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'ResizePreprocessor',
            'width': self.width,
            'height': self.height,
            'interpolation': self.interpolation
        }

class ResizeWithAspectPreprocessor(FramePreprocessor):
    """Resize while maintaining aspect ratio with padding."""
    
    def __init__(self, width: int, height: int, pad_color: Tuple[int, int, int] = (0, 0, 0)):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
            
        self.width = width
        self.height = height
        self.pad_color = pad_color
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Calculate scale to fit within target dimensions
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Calculate padding
        pad_w = (self.width - new_w) // 2
        pad_h = (self.height - new_h) // 2
        
        # Create padded image
        if len(frame.shape) == 3:
            padded = np.full((self.height, self.width, frame.shape[2]), self.pad_color, dtype=frame.dtype)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        else:
            padded = np.full((self.height, self.width), self.pad_color[0], dtype=frame.dtype)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return padded
    
    def get_name(self) -> str:
        return "resize_with_aspect"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'ResizeWithAspectPreprocessor',
            'width': self.width,
            'height': self.height,
            'pad_color': self.pad_color
        }

class BlackMaskPreprocessor(FramePreprocessor):
    """Apply black masks to frame regions."""
    
    def __init__(self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0):
        if any(x < 0 for x in [left, right, top, bottom]):
            raise ValueError("Mask sizes must be non-negative")
            
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Apply masks with bounds checking
        if self.left > 0:
            frame[:, :min(self.left, w)] = 0
        if self.right > 0:
            frame[:, max(0, w - self.right):] = 0
        if self.top > 0:
            frame[:min(self.top, h), :] = 0
        if self.bottom > 0:
            frame[max(0, h - self.bottom):, :] = 0
            
        return frame
    
    def get_name(self) -> str:
        return "black_mask"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'BlackMaskPreprocessor',
            'left': self.left, 'right': self.right,
            'top': self.top, 'bottom': self.bottom
        }

class RotatePreprocessor(FramePreprocessor):
    """Rotate frame by specified angle."""
    
    def __init__(self, angle: float, expand: bool = True):
        self.angle = angle
        self.expand = expand
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        
        if self.expand:
            # Calculate new dimensions
            cos_angle = abs(M[0, 0])
            sin_angle = abs(M[0, 1])
            new_w = int(h * sin_angle + w * cos_angle)
            new_h = int(h * cos_angle + w * sin_angle)
            
            # Adjust translation
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            
            return cv2.warpAffine(frame, M, (new_w, new_h))
        else:
            return cv2.warpAffine(frame, M, (w, h))
    
    def get_name(self) -> str:
        return "rotate"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'RotatePreprocessor',
            'angle': self.angle,
            'expand': self.expand
        }