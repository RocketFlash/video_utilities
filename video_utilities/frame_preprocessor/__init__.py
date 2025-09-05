from .base import FramePreprocessor
from .composer import PreprocessorComposer, ConditionalPreprocessor
from .geometric import (
    CropPreprocessor,
    ResizePreprocessor,
    ResizeWithAspectPreprocessor,
    BlackMaskPreprocessor,
    RotatePreprocessor
)
from .color import (
    BrightnessContrastPreprocessor,
    CLAHEPreprocessor,
    GammaPreprocessor,
    HSVAdjustPreprocessor,
    GrayscalePreprocessor
)
from .noise import (
    GaussianBlurPreprocessor,
    MedianBlurPreprocessor,
    BilateralFilterPreprocessor
)
from .custom import (
    FunctionPreprocessor,
    LambdaPreprocessor,
    NoOpPreprocessor
)

__all__ = [
    # Base
    'FramePreprocessor',
    'PreprocessorComposer',
    'ConditionalPreprocessor',
    
    # Geometric
    'CropPreprocessor',
    'ResizePreprocessor', 
    'ResizeWithAspectPreprocessor',
    'BlackMaskPreprocessor',
    'RotatePreprocessor',
    
    # Color
    'BrightnessContrastPreprocessor',
    'CLAHEPreprocessor',
    'GammaPreprocessor',
    'HSVAdjustPreprocessor',
    'GrayscalePreprocessor',
    
    # Noise
    'GaussianBlurPreprocessor',
    'MedianBlurPreprocessor',
    'BilateralFilterPreprocessor',
    
    # Custom
    'FunctionPreprocessor',
    'LambdaPreprocessor',
    'NoOpPreprocessor'
]

# Convenience factory functions
def create_basic_pipeline():
    """Create basic preprocessing pipeline."""
    return PreprocessorComposer([
        ResizePreprocessor(640, 480),
        BrightnessContrastPreprocessor(brightness=0, contrast=1.0)
    ], name="basic")

def create_enhancement_pipeline():
    """Create enhancement preprocessing pipeline."""
    return PreprocessorComposer([
        CLAHEPreprocessor(clip_limit=2.0),
        BrightnessContrastPreprocessor(brightness=10, contrast=1.1),
        GaussianBlurPreprocessor(kernel_size=3, sigma=0.5)
    ], name="enhancement")

def create_black_mask_pipeline(left=0, right=0, top=0, bottom=0):
    """Create black mask pipeline (backward compatibility)."""
    return PreprocessorComposer([
        BlackMaskPreprocessor(left, right, top, bottom)
    ], name="black_mask")