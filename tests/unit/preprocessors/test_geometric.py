import pytest
import numpy as np
import cv2
from video_utilities.frame_preprocessor import *

class TestCropPreprocessor:
    
    def test_crop_basic(self, sample_image):
        """Test basic cropping functionality."""
        crop = CropPreprocessor(100, 100, 400, 300)
        result = crop(sample_image)
        
        assert result.shape[:2] == (200, 300)  # height, width
    
    def test_crop_invalid_coordinates(self):
        """Test that invalid crop coordinates raise error."""
        with pytest.raises(ValueError):
            CropPreprocessor(400, 100, 100, 300)  # x1 > x2
        
        with pytest.raises(ValueError):
            CropPreprocessor(100, 300, 400, 100)  # y1 > y2
    
    def test_crop_bounds_checking(self, sample_image):
        """Test that crop handles out-of-bounds coordinates."""
        h, w = sample_image.shape[:2]
        
        # Coordinates larger than image
        crop = CropPreprocessor(-10, -10, w + 100, h + 100)
        result = crop(sample_image)
        
        # Should crop to valid bounds
        assert result.shape[:2] == sample_image.shape[:2]

class TestResizePreprocessor:
    
    def test_resize_basic(self, sample_image):
        """Test basic resize functionality."""
        resize = ResizePreprocessor(320, 240)
        result = resize(sample_image)
        
        assert result.shape[:2] == (240, 320)
    
    def test_resize_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            ResizePreprocessor(0, 240)
        
        with pytest.raises(ValueError):
            ResizePreprocessor(320, -240)

class TestResizeWithAspectPreprocessor:
    
    def test_resize_with_aspect(self, sample_image):
        """Test resize with aspect ratio preservation."""
        resize_aspect = ResizeWithAspectPreprocessor(400, 400, (128, 128, 128))
        result = resize_aspect(sample_image)
        
        assert result.shape[:2] == (400, 400)
        
        # Check that padding was added (gray pixels should exist)
        gray_pixels = np.all(result == [128, 128, 128], axis=2)
        assert np.any(gray_pixels)

class TestBlackMaskPreprocessor:
    
    def test_black_mask_basic(self, sample_image):
        """Test basic black mask functionality."""
        mask = BlackMaskPreprocessor(left=50, right=50, top=30, bottom=30)
        result = mask(sample_image)
        
        # Check that edges are black
        assert np.all(result[:, :50] == 0)  # Left
        assert np.all(result[:, -50:] == 0)  # Right
        assert np.all(result[:30, :] == 0)  # Top
        assert np.all(result[-30:, :] == 0)  # Bottom
    
    def test_black_mask_invalid_values(self):
        """Test that negative mask values raise error."""
        with pytest.raises(ValueError):
            BlackMaskPreprocessor(left=-10)

class TestRotatePreprocessor:
    
    def test_rotate_basic(self, sample_image):
        """Test basic rotation functionality."""
        rotate = RotatePreprocessor(45, expand=True)
        result = rotate(sample_image)
        
        # Rotated image should be larger due to expansion
        assert result.shape[0] > sample_image.shape[0]
        assert result.shape[1] > sample_image.shape[1]
    
    def test_rotate_no_expand(self, sample_image):
        """Test rotation without expansion."""
        rotate = RotatePreprocessor(45, expand=False)
        result = rotate(sample_image)
        
        # Should maintain original dimensions
        assert result.shape[:2] == sample_image.shape[:2]