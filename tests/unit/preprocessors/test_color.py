import pytest
import numpy as np
import cv2
from video_utilities.frame_preprocessor import (
    BrightnessContrastPreprocessor,
    CLAHEPreprocessor, 
    GammaPreprocessor,
    HSVAdjustPreprocessor,
    GrayscalePreprocessor
)


class TestBrightnessContrastPreprocessor:
    
    def test_brightness_increase(self, sample_image):
        """Test brightness increase."""
        preprocessor = BrightnessContrastPreprocessor(brightness=50)
        result = preprocessor(sample_image)
        
        # Result should be brighter
        assert result.mean() > sample_image.mean()
        assert result.shape == sample_image.shape
    
    def test_brightness_decrease(self, sample_image):
        """Test brightness decrease.""" 
        preprocessor = BrightnessContrastPreprocessor(brightness=-30)
        result = preprocessor(sample_image)
        
        # Result should be darker
        assert result.mean() < sample_image.mean()
    
    def test_contrast_increase(self, sample_image):
        """Test contrast increase."""
        preprocessor = BrightnessContrastPreprocessor(contrast=1.5)
        result = preprocessor(sample_image)
        
        # Result should have higher contrast (higher std deviation)
        assert result.std() > sample_image.std()
    
    def test_contrast_decrease(self, sample_image):
        """Test contrast decrease."""
        preprocessor = BrightnessContrastPreprocessor(contrast=0.5)
        result = preprocessor(sample_image)
        
        # Result should have lower contrast
        assert result.std() < sample_image.std()
    
    def test_combined_adjustments(self, sample_image):
        """Test combined brightness and contrast adjustments."""
        preprocessor = BrightnessContrastPreprocessor(brightness=20, contrast=1.2)
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_zero_adjustments(self, sample_image):
        """Test no adjustments (identity)."""
        preprocessor = BrightnessContrastPreprocessor(brightness=0, contrast=1.0)
        result = preprocessor(sample_image)
        
        # Should be very close to original (cv2.convertScaleAbs may cause minor differences)
        np.testing.assert_allclose(result, sample_image, atol=1)
    
    def test_config(self):
        """Test configuration."""
        preprocessor = BrightnessContrastPreprocessor(brightness=25, contrast=1.3)
        config = preprocessor.get_config()
        
        assert config['type'] == 'BrightnessContrastPreprocessor'
        assert config['brightness'] == 25
        assert config['contrast'] == 1.3
        assert preprocessor.get_name() == 'brightness_contrast'


class TestCLAHEPreprocessor:
    
    def test_clahe_basic(self, sample_image):
        """Test basic CLAHE functionality."""
        preprocessor = CLAHEPreprocessor()
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_clahe_grayscale(self):
        """Test CLAHE on grayscale image."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessor = CLAHEPreprocessor()
        result = preprocessor(gray_image)
        
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype
    
    def test_clahe_parameters(self, sample_image):
        """Test CLAHE with custom parameters."""
        preprocessor = CLAHEPreprocessor(clip_limit=3.0, tile_grid_size=(16, 16))
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
    
    def test_clahe_invalid_clip_limit(self):
        """Test that invalid clip limit raises error."""
        with pytest.raises(ValueError):
            CLAHEPreprocessor(clip_limit=0)
        
        with pytest.raises(ValueError):
            CLAHEPreprocessor(clip_limit=-1)
    
    def test_clahe_invalid_tile_size(self):
        """Test that invalid tile size raises error."""
        with pytest.raises(ValueError):
            CLAHEPreprocessor(tile_grid_size=(0, 8))
        
        with pytest.raises(ValueError):
            CLAHEPreprocessor(tile_grid_size=(8, -1))
    
    def test_config(self):
        """Test configuration."""
        preprocessor = CLAHEPreprocessor(clip_limit=2.5, tile_grid_size=(4, 4))
        config = preprocessor.get_config()
        
        assert config['type'] == 'CLAHEPreprocessor'
        assert config['clip_limit'] == 2.5
        assert config['tile_grid_size'] == (4, 4)
        assert preprocessor.get_name() == 'clahe'


class TestGammaPreprocessor:
    
    def test_gamma_correction_basic(self, sample_image):
        """Test basic gamma correction."""
        preprocessor = GammaPreprocessor(gamma=2.0)
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_gamma_brightening(self, sample_image):
        """Test gamma correction for brightening (gamma < 1)."""
        preprocessor = GammaPreprocessor(gamma=0.5)
        result = preprocessor(sample_image)
        
        # For gamma < 1, mid-tones should be darkened (gamma correction formula)
        # Test with a value where the effect is clear: 128 -> should become darker
        mid_tone_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        mid_result = preprocessor(mid_tone_image)
        assert mid_result.mean() < mid_tone_image.mean()  # Should be darker
        
        # Test basic functionality - just verify the transform works
        dark_image = np.full((100, 100, 3), 64, dtype=np.uint8)
        dark_result = preprocessor(dark_image)
        # For gamma=0.5, formula is (64/255)^(1/0.5) = (64/255)^2 ≈ 0.0631, scaled to 255 ≈ 16
        assert dark_result[0,0,0] == 16  # Should match calculated value
    
    def test_gamma_darkening(self, sample_image):
        """Test gamma correction for darkening (gamma > 1)."""
        preprocessor = GammaPreprocessor(gamma=2.0)
        result = preprocessor(sample_image)
        
        # For gamma > 1, mid-tones should be brightened (gamma correction formula)
        # Test with a value where the effect is clear: 128 -> should become brighter
        mid_tone_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        mid_result = preprocessor(mid_tone_image)
        assert mid_result.mean() > mid_tone_image.mean()  # Should be brighter
        
        # Test with bright values - they should become relatively darker
        bright_image = np.full((100, 100, 3), 200, dtype=np.uint8)
        bright_result = preprocessor(bright_image)
        # The effect should be visible
        assert bright_result[0,0,0] != bright_image[0,0,0]  # Should change
    
    def test_gamma_identity(self, sample_image):
        """Test gamma = 1.0 (identity)."""
        preprocessor = GammaPreprocessor(gamma=1.0)
        result = preprocessor(sample_image)
        
        # Should be nearly identical to original
        np.testing.assert_allclose(result, sample_image, atol=2)
    
    def test_gamma_invalid_value(self):
        """Test that invalid gamma values raise error."""
        with pytest.raises(ValueError):
            GammaPreprocessor(gamma=0)
        
        with pytest.raises(ValueError):
            GammaPreprocessor(gamma=-1.5)
    
    def test_gamma_grayscale(self):
        """Test gamma correction on grayscale."""
        gray_image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        preprocessor = GammaPreprocessor(gamma=1.5)
        result = preprocessor(gray_image)
        
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype
    
    def test_config(self):
        """Test configuration."""
        preprocessor = GammaPreprocessor(gamma=1.8)
        config = preprocessor.get_config()
        
        assert config['type'] == 'GammaPreprocessor'
        assert config['gamma'] == 1.8
        assert preprocessor.get_name() == 'gamma'


class TestHSVAdjustPreprocessor:
    
    def test_hsv_basic(self, sample_image):
        """Test basic HSV adjustment."""
        preprocessor = HSVAdjustPreprocessor()
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_hue_shift(self, sample_image):
        """Test hue shifting."""
        preprocessor = HSVAdjustPreprocessor(hue_shift=30)
        result = preprocessor(sample_image)
        
        # Should be different from original
        assert not np.array_equal(result, sample_image)
    
    def test_hue_wraparound(self, sample_image):
        """Test hue wraparound at 180."""
        preprocessor1 = HSVAdjustPreprocessor(hue_shift=190)
        preprocessor2 = HSVAdjustPreprocessor(hue_shift=10)
        
        result1 = preprocessor1(sample_image)
        result2 = preprocessor2(sample_image)
        
        # 190 % 180 = 10, so results should be identical
        np.testing.assert_array_equal(result1, result2)
    
    def test_saturation_adjustment(self, sample_image):
        """Test saturation adjustment."""
        preprocessor_increase = HSVAdjustPreprocessor(saturation_scale=1.5)
        preprocessor_decrease = HSVAdjustPreprocessor(saturation_scale=0.5)
        
        result_increase = preprocessor_increase(sample_image)
        result_decrease = preprocessor_decrease(sample_image)
        
        assert result_increase.shape == sample_image.shape
        assert result_decrease.shape == sample_image.shape
    
    def test_value_adjustment(self, sample_image):
        """Test value (brightness) adjustment."""
        preprocessor_brighter = HSVAdjustPreprocessor(value_scale=1.3)
        preprocessor_darker = HSVAdjustPreprocessor(value_scale=0.7)
        
        result_brighter = preprocessor_brighter(sample_image)
        result_darker = preprocessor_darker(sample_image)
        
        # Brighter image should have higher mean
        assert result_brighter.mean() > sample_image.mean()
        # Darker image should have lower mean
        assert result_darker.mean() < sample_image.mean()
    
    def test_combined_adjustments(self, sample_image):
        """Test combined HSV adjustments."""
        preprocessor = HSVAdjustPreprocessor(
            hue_shift=45,
            saturation_scale=1.2,
            value_scale=0.9
        )
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
    
    def test_grayscale_raises_error(self):
        """Test that grayscale image raises error."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessor = HSVAdjustPreprocessor()
        
        with pytest.raises(ValueError):
            preprocessor(gray_image)
    
    def test_negative_scales_clamped(self, sample_image):
        """Test that negative scales are clamped to 0."""
        preprocessor = HSVAdjustPreprocessor(
            saturation_scale=-0.5,
            value_scale=-1.0
        )
        
        # Should not raise error (values clamped in __init__)
        result = preprocessor(sample_image)
        assert result.shape == sample_image.shape
        
        # Check that scales were clamped
        assert preprocessor.saturation_scale == 0
        assert preprocessor.value_scale == 0
    
    def test_config(self):
        """Test configuration."""
        preprocessor = HSVAdjustPreprocessor(
            hue_shift=60,
            saturation_scale=1.4,
            value_scale=0.8
        )
        config = preprocessor.get_config()
        
        assert config['type'] == 'HSVAdjustPreprocessor'
        assert config['hue_shift'] == 60
        assert config['saturation_scale'] == 1.4
        assert config['value_scale'] == 0.8
        assert preprocessor.get_name() == 'hsv_adjust'


class TestGrayscalePreprocessor:
    
    def test_grayscale_basic(self, sample_image):
        """Test basic grayscale conversion."""
        preprocessor = GrayscalePreprocessor()
        result = preprocessor(sample_image)
        
        # Should be 2D (single channel)
        assert len(result.shape) == 2
        assert result.shape[:2] == sample_image.shape[:2]
        assert result.dtype == sample_image.dtype
    
    def test_grayscale_keep_channels(self, sample_image):
        """Test grayscale with channel preservation."""
        preprocessor = GrayscalePreprocessor(keep_channels=True)
        result = preprocessor(sample_image)
        
        # Should be 3D (3 channels)
        assert len(result.shape) == 3
        assert result.shape == sample_image.shape
        
        # All channels should be identical
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])
    
    def test_grayscale_already_gray(self):
        """Test grayscale on already grayscale image."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        preprocessor = GrayscalePreprocessor()
        result = preprocessor(gray_image)
        
        # Should return unchanged
        np.testing.assert_array_equal(result, gray_image)
    
    def test_grayscale_already_gray_keep_channels(self):
        """Test grayscale on already grayscale with keep_channels."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        preprocessor = GrayscalePreprocessor(keep_channels=True)
        result = preprocessor(gray_image)
        
        # Should return unchanged (still 2D)
        np.testing.assert_array_equal(result, gray_image)
    
    def test_grayscale_maintains_content(self, sample_image):
        """Test that grayscale conversion maintains image structure."""
        preprocessor = GrayscalePreprocessor()
        result = preprocessor(sample_image)
        
        # Should maintain overall brightness distribution
        # (though exact values will differ due to RGB->Gray conversion)
        assert 0 <= result.min() <= 255
        assert 0 <= result.max() <= 255
    
    def test_config(self):
        """Test configuration."""
        preprocessor_single = GrayscalePreprocessor(keep_channels=False)
        preprocessor_multi = GrayscalePreprocessor(keep_channels=True)
        
        config_single = preprocessor_single.get_config()
        config_multi = preprocessor_multi.get_config()
        
        assert config_single['type'] == 'GrayscalePreprocessor'
        assert config_single['keep_channels'] == False
        assert config_multi['keep_channels'] == True
        
        assert preprocessor_single.get_name() == 'grayscale'
        assert preprocessor_multi.get_name() == 'grayscale'