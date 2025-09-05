"""
Visual tests for preprocessors - generates images for manual inspection.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from video_utilities.frame_preprocessor import *

class TestVisualPreprocessors:
    """Visual tests that save processed images for manual inspection."""
    
    def test_geometric_preprocessors(self, sample_image, test_results_dir):
        """Test geometric preprocessors and save results."""
        output_dir = test_results_dir / "geometric"
        output_dir.mkdir(exist_ok=True)
        
        # Original
        self._save_image(sample_image, output_dir / "00_original.jpg")
        
        # Crop
        crop = CropPreprocessor(100, 100, 500, 350)
        cropped = crop(sample_image)
        self._save_image(cropped, output_dir / "01_cropped.jpg")
        
        # Resize
        resize = ResizePreprocessor(320, 240)
        resized = resize(sample_image)
        self._save_image(resized, output_dir / "02_resized.jpg")
        
        # Resize with aspect
        resize_aspect = ResizeWithAspectPreprocessor(400, 400, (128, 128, 128))
        resized_aspect = resize_aspect(sample_image)
        self._save_image(resized_aspect, output_dir / "03_resized_aspect.jpg")
        
        # Black mask
        black_mask = BlackMaskPreprocessor(50, 50, 30, 30)
        masked = black_mask(sample_image)
        self._save_image(masked, output_dir / "04_black_mask.jpg")
        
        # Rotate
        rotate = RotatePreprocessor(15, expand=True)
        rotated = rotate(sample_image)
        self._save_image(rotated, output_dir / "05_rotated.jpg")
    
    def test_color_preprocessors(self, sample_image, test_results_dir):
        """Test color preprocessors and save results."""
        output_dir = test_results_dir / "color"
        output_dir.mkdir(exist_ok=True)
        
        # Original
        self._save_image(sample_image, output_dir / "00_original.jpg")
        
        # Brightness/Contrast
        bright_contrast = BrightnessContrastPreprocessor(brightness=30, contrast=1.2)
        enhanced = bright_contrast(sample_image)
        self._save_image(enhanced, output_dir / "01_brightness_contrast.jpg")
        
        # CLAHE
        clahe = CLAHEPreprocessor(clip_limit=3.0)
        clahe_enhanced = clahe(sample_image)
        self._save_image(clahe_enhanced, output_dir / "02_clahe.jpg")
        
        # Gamma
        gamma = GammaPreprocessor(gamma=0.7)
        gamma_corrected = gamma(sample_image)
        self._save_image(gamma_corrected, output_dir / "03_gamma.jpg")
        
        # HSV adjust
        hsv = HSVAdjustPreprocessor(hue_shift=30, saturation_scale=1.3, value_scale=1.1)
        hsv_adjusted = hsv(sample_image)
        self._save_image(hsv_adjusted, output_dir / "04_hsv_adjusted.jpg")
        
        # Grayscale
        gray = GrayscalePreprocessor(keep_channels=True)
        grayscale = gray(sample_image)
        self._save_image(grayscale, output_dir / "05_grayscale.jpg")
    
    def test_noise_preprocessors(self, sample_image, test_results_dir):
        """Test noise and blur preprocessors."""
        output_dir = test_results_dir / "noise"
        output_dir.mkdir(exist_ok=True)
        
        # Original
        self._save_image(sample_image, output_dir / "00_original.jpg")
        
        # Gaussian blur
        gaussian = GaussianBlurPreprocessor(kernel_size=7, sigma=2.0)
        blurred = gaussian(sample_image)
        self._save_image(blurred, output_dir / "01_gaussian_blur.jpg")
        
        # Median blur
        median = MedianBlurPreprocessor(kernel_size=5)
        median_blurred = median(sample_image)
        self._save_image(median_blurred, output_dir / "02_median_blur.jpg")
        
        # Bilateral filter
        bilateral = BilateralFilterPreprocessor(d=9, sigma_color=75, sigma_space=75)
        bilateral_filtered = bilateral(sample_image)
        self._save_image(bilateral_filtered, output_dir / "03_bilateral_filter.jpg")
    
    def test_composed_pipeline(self, sample_image, test_results_dir):
        """Test composition of multiple preprocessors."""
        output_dir = test_results_dir / "composed"
        output_dir.mkdir(exist_ok=True)
        
        # Original
        self._save_image(sample_image, output_dir / "00_original.jpg")
        
        # Enhancement pipeline
        pipeline = PreprocessorComposer([
            CropPreprocessor(50, 50, 590, 430),
            ResizePreprocessor(512, 384),
            CLAHEPreprocessor(clip_limit=2.0),
            BrightnessContrastPreprocessor(brightness=10, contrast=1.1),
            GaussianBlurPreprocessor(kernel_size=3, sigma=0.5)
        ], name="enhancement")
        
        # Save intermediate results
        intermediate = sample_image
        for i, preprocessor in enumerate(pipeline.preprocessors):
            intermediate = preprocessor(intermediate)
            self._save_image(intermediate, output_dir / f"{i+1:02d}_{preprocessor.get_name()}.jpg")
        
        # Final result
        final = pipeline(sample_image)
        self._save_image(final, output_dir / "99_final_composed.jpg")
    
    def test_conditional_preprocessing(self, test_results_dir):
        """Test conditional preprocessing with different image conditions."""
        output_dir = test_results_dir / "conditional"
        output_dir.mkdir(exist_ok=True)
        
        # Create dark and bright test images
        dark_image = create_test_image()
        dark_image = (dark_image * 0.3).astype(np.uint8)  # Make it dark
        
        bright_image = create_test_image()
        bright_image = np.clip(bright_image * 1.5, 0, 255).astype(np.uint8)  # Make it bright
        
        # Condition function
        def is_dark(frame):
            return frame.mean() < 100
        
        # Conditional enhancer
        enhancer = ConditionalPreprocessor(
            CLAHEPreprocessor(clip_limit=3.0),
            is_dark,
            "dark_enhancement"
        )
        
        # Test on dark image
        self._save_image(dark_image, output_dir / "01_dark_original.jpg")
        enhanced_dark = enhancer(dark_image)
        self._save_image(enhanced_dark, output_dir / "02_dark_enhanced.jpg")
        
        # Test on bright image
        self._save_image(bright_image, output_dir / "03_bright_original.jpg")
        enhanced_bright = enhancer(bright_image)  # Should remain unchanged
        self._save_image(enhanced_bright, output_dir / "04_bright_unchanged.jpg")
    
    def test_comparison_grid(self, sample_image, test_results_dir):
        """Create a comparison grid of different preprocessing effects."""
        output_dir = test_results_dir / "comparison"
        output_dir.mkdir(exist_ok=True)
        
        # Define preprocessors to compare
        preprocessors = [
            ("Original", NoOpPreprocessor()),
            ("Crop", CropPreprocessor(80, 80, 560, 400)),
            ("Bright+", BrightnessContrastPreprocessor(brightness=30, contrast=1.2)),
            ("CLAHE", CLAHEPreprocessor(clip_limit=2.5)),
            ("Gamma", GammaPreprocessor(gamma=0.8)),
            ("Blur", GaussianBlurPreprocessor(kernel_size=5, sigma=1.5)),
            ("HSV", HSVAdjustPreprocessor(hue_shift=20, saturation_scale=1.3)),
            ("Grayscale", GrayscalePreprocessor(keep_channels=True))
        ]
        
        # Create comparison grid
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (name, preprocessor) in enumerate(preprocessors):
            processed = preprocessor(sample_image)
            axes[i].imshow(processed)
            axes[i].set_title(name)
            axes[i].axis('off')
            
            # Also save individual images
            self._save_image(processed, output_dir / f"{i:02d}_{name.lower().replace('+', 'plus')}.jpg")
        
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_image(self, image, path):
        """Save image to path, handling RGB->BGR conversion."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        cv2.imwrite(str(path), image_bgr)

def create_test_image(width=640, height=480):
    """Create a synthetic test image with various patterns."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(height):
        image[i, :, 0] = int(255 * i / height)
    
    for j in range(width):
        image[:, j, 1] = int(255 * j / width)
    
    # Add geometric shapes
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(image, (300, 200), 50, (255, 0, 0), -1)
    cv2.line(image, (400, 100), (500, 300), (0, 255, 0), 5)
    
    # Add text
    cv2.putText(image, "TEST", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image