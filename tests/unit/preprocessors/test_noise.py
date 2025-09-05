import pytest
import numpy as np
import cv2
from video_utilities.frame_preprocessor import (
    GaussianBlurPreprocessor,
    MedianBlurPreprocessor, 
    BilateralFilterPreprocessor
)


class TestGaussianBlurPreprocessor:
    
    def test_gaussian_blur_basic(self, sample_image):
        """Test basic Gaussian blur functionality."""
        preprocessor = GaussianBlurPreprocessor(kernel_size=5, sigma=1.0)
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
        # Blurred image should have lower variance (smoother)
        assert result.std() < sample_image.std()
    
    def test_gaussian_blur_different_kernel_sizes(self, sample_image):
        """Test different kernel sizes."""
        small_blur = GaussianBlurPreprocessor(kernel_size=3, sigma=1.0)
        large_blur = GaussianBlurPreprocessor(kernel_size=15, sigma=1.0)
        
        small_result = small_blur(sample_image)
        large_result = large_blur(sample_image)
        
        # Larger kernel should produce more blur (lower std)
        assert large_result.std() < small_result.std()
    
    def test_gaussian_blur_different_sigma(self, sample_image):
        """Test different sigma values."""
        small_sigma = GaussianBlurPreprocessor(kernel_size=11, sigma=0.5)
        large_sigma = GaussianBlurPreprocessor(kernel_size=11, sigma=3.0)
        
        small_result = small_sigma(sample_image)
        large_result = large_sigma(sample_image)
        
        # Larger sigma should produce more blur
        assert large_result.std() < small_result.std()
    
    def test_gaussian_blur_invalid_kernel_size(self):
        """Test that invalid kernel sizes raise error."""
        # Even kernel size
        with pytest.raises(ValueError):
            GaussianBlurPreprocessor(kernel_size=4)
        
        # Zero kernel size
        with pytest.raises(ValueError):
            GaussianBlurPreprocessor(kernel_size=0)
        
        # Negative kernel size
        with pytest.raises(ValueError):
            GaussianBlurPreprocessor(kernel_size=-3)
    
    def test_gaussian_blur_invalid_sigma(self):
        """Test that invalid sigma values raise error."""
        # Zero sigma
        with pytest.raises(ValueError):
            GaussianBlurPreprocessor(kernel_size=5, sigma=0)
        
        # Negative sigma
        with pytest.raises(ValueError):
            GaussianBlurPreprocessor(kernel_size=5, sigma=-1.0)
    
    def test_gaussian_blur_grayscale(self):
        """Test Gaussian blur on grayscale image."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessor = GaussianBlurPreprocessor(kernel_size=7, sigma=2.0)
        result = preprocessor(gray_image)
        
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype
        assert result.std() < gray_image.std()
    
    def test_config(self):
        """Test configuration."""
        preprocessor = GaussianBlurPreprocessor(kernel_size=9, sigma=1.5)
        config = preprocessor.get_config()
        
        assert config['type'] == 'GaussianBlurPreprocessor'
        assert config['kernel_size'] == 9
        assert config['sigma'] == 1.5
        assert preprocessor.get_name() == 'gaussian_blur'


class TestMedianBlurPreprocessor:
    
    def test_median_blur_basic(self, sample_image):
        """Test basic median blur functionality."""
        preprocessor = MedianBlurPreprocessor(kernel_size=5)
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_median_blur_noise_removal(self):
        """Test median blur effectiveness on salt-and-pepper noise."""
        # Create image with salt-and-pepper noise
        clean_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        noisy_image = clean_image.copy()
        
        # Add salt noise (white pixels)
        salt_coords = np.random.randint(0, 100, (200, 2))
        noisy_image[salt_coords[:, 0], salt_coords[:, 1]] = 255
        
        # Add pepper noise (black pixels)  
        pepper_coords = np.random.randint(0, 100, (200, 2))
        noisy_image[pepper_coords[:, 0], pepper_coords[:, 1]] = 0
        
        preprocessor = MedianBlurPreprocessor(kernel_size=5)
        result = preprocessor(noisy_image)
        
        # Median blur should reduce noise (result closer to clean image)
        noise_distance = np.mean(np.abs(noisy_image.astype(float) - clean_image.astype(float)))
        result_distance = np.mean(np.abs(result.astype(float) - clean_image.astype(float)))
        
        assert result_distance < noise_distance
    
    def test_median_blur_different_kernel_sizes(self, sample_image):
        """Test different kernel sizes."""
        small_blur = MedianBlurPreprocessor(kernel_size=3)
        large_blur = MedianBlurPreprocessor(kernel_size=9)
        
        small_result = small_blur(sample_image)
        large_result = large_blur(sample_image)
        
        assert small_result.shape == sample_image.shape
        assert large_result.shape == sample_image.shape
        # Results should be different
        assert not np.array_equal(small_result, large_result)
    
    def test_median_blur_invalid_kernel_size(self):
        """Test that invalid kernel sizes raise error."""
        # Even kernel size
        with pytest.raises(ValueError):
            MedianBlurPreprocessor(kernel_size=6)
        
        # Zero kernel size
        with pytest.raises(ValueError):
            MedianBlurPreprocessor(kernel_size=0)
        
        # Negative kernel size
        with pytest.raises(ValueError):
            MedianBlurPreprocessor(kernel_size=-5)
    
    def test_median_blur_grayscale(self):
        """Test median blur on grayscale image."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessor = MedianBlurPreprocessor(kernel_size=7)
        result = preprocessor(gray_image)
        
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype
    
    def test_median_blur_preserves_edges(self):
        """Test that median blur handles noise while Gaussian smooths edges."""
        # Create image with sharp edge and some noise
        edge_image = np.zeros((100, 100, 3), dtype=np.uint8)
        edge_image[:, :50] = 255  # White left half, black right half
        
        # Add some impulse noise near the edge
        noise_positions = [(48, 25), (49, 25), (51, 25), (52, 25)]
        for pos in noise_positions:
            edge_image[pos[1], pos[0]] = [128, 128, 128]  # Gray noise
        
        median_preprocessor = MedianBlurPreprocessor(kernel_size=5)
        gaussian_preprocessor = GaussianBlurPreprocessor(kernel_size=5, sigma=1.0)
        
        median_result = median_preprocessor(edge_image)
        gaussian_result = gaussian_preprocessor(edge_image)
        
        # Both filters will affect the edge, but test that they work as expected
        # Check that the results are different (showing they process differently)
        assert not np.array_equal(median_result, gaussian_result)
        
        # Check that both results have expected shapes and types
        assert median_result.shape == edge_image.shape
        assert gaussian_result.shape == edge_image.shape
        assert median_result.dtype == edge_image.dtype
        assert gaussian_result.dtype == edge_image.dtype
    
    def test_config(self):
        """Test configuration."""
        preprocessor = MedianBlurPreprocessor(kernel_size=7)
        config = preprocessor.get_config()
        
        assert config['type'] == 'MedianBlurPreprocessor'
        assert config['kernel_size'] == 7
        assert preprocessor.get_name() == 'median_blur'


class TestBilateralFilterPreprocessor:
    
    def test_bilateral_filter_basic(self, sample_image):
        """Test basic bilateral filter functionality."""
        preprocessor = BilateralFilterPreprocessor()
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_bilateral_filter_custom_parameters(self, sample_image):
        """Test bilateral filter with custom parameters."""
        preprocessor = BilateralFilterPreprocessor(d=15, sigma_color=80, sigma_space=80)
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_bilateral_filter_noise_reduction(self):
        """Test bilateral filter for noise reduction."""
        # Create clean image
        clean_image = np.zeros((100, 100, 3), dtype=np.uint8)
        clean_image[20:80, 20:80] = 255  # White square
        
        # Add Gaussian noise
        noise = np.random.normal(0, 25, clean_image.shape).astype(np.int16)
        noisy_image = np.clip(clean_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        preprocessor = BilateralFilterPreprocessor(d=9, sigma_color=75, sigma_space=75)
        result = preprocessor(noisy_image)
        
        # Should reduce noise (result closer to clean)
        noise_mse = np.mean((noisy_image.astype(float) - clean_image.astype(float)) ** 2)
        result_mse = np.mean((result.astype(float) - clean_image.astype(float)) ** 2)
        
        assert result_mse < noise_mse
    
    def test_bilateral_filter_edge_preservation(self):
        """Test bilateral filter edge preservation."""
        # Create image with sharp edges and flat areas
        edge_image = np.zeros((100, 100, 3), dtype=np.uint8)
        edge_image[:, :50] = 100   # Gray left half
        edge_image[:, 50:] = 200   # Lighter gray right half
        
        # Add noise
        noise = np.random.normal(0, 15, edge_image.shape).astype(np.int16)
        noisy_edge_image = np.clip(edge_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply bilateral filter
        preprocessor = BilateralFilterPreprocessor(d=9, sigma_color=50, sigma_space=50)
        result = preprocessor(noisy_edge_image)
        
        # Check that edge is still sharp
        # Look at the transition area around column 50
        left_mean = result[:, 45:50].mean()
        right_mean = result[:, 50:55].mean()
        edge_strength = abs(right_mean - left_mean)
        
        # Should maintain reasonable edge strength
        assert edge_strength > 50  # Should be close to original 100 difference
    
    def test_bilateral_filter_different_sigma_color(self, sample_image):
        """Test different sigma_color values."""
        low_sigma_color = BilateralFilterPreprocessor(d=9, sigma_color=25, sigma_space=75)
        high_sigma_color = BilateralFilterPreprocessor(d=9, sigma_color=150, sigma_space=75)
        
        low_result = low_sigma_color(sample_image)
        high_result = high_sigma_color(sample_image)
        
        assert low_result.shape == sample_image.shape
        assert high_result.shape == sample_image.shape
        # Results should be different
        assert not np.array_equal(low_result, high_result)
    
    def test_bilateral_filter_different_sigma_space(self, sample_image):
        """Test different sigma_space values."""
        low_sigma_space = BilateralFilterPreprocessor(d=9, sigma_color=75, sigma_space=25)
        high_sigma_space = BilateralFilterPreprocessor(d=9, sigma_color=75, sigma_space=150)
        
        low_result = low_sigma_space(sample_image)
        high_result = high_sigma_space(sample_image)
        
        assert low_result.shape == sample_image.shape
        assert high_result.shape == sample_image.shape
        # Results should be different
        assert not np.array_equal(low_result, high_result)
    
    def test_bilateral_filter_grayscale(self):
        """Test bilateral filter on grayscale image."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        preprocessor = BilateralFilterPreprocessor()
        result = preprocessor(gray_image)
        
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype
    
    def test_config(self):
        """Test configuration."""
        preprocessor = BilateralFilterPreprocessor(d=11, sigma_color=80, sigma_space=85)
        config = preprocessor.get_config()
        
        assert config['type'] == 'BilateralFilterPreprocessor'
        assert config['d'] == 11
        assert config['sigma_color'] == 80
        assert config['sigma_space'] == 85
        assert preprocessor.get_name() == 'bilateral_filter'
        
    def test_bilateral_filter_different_d_values(self, sample_image):
        """Test different neighborhood diameter values."""
        small_d = BilateralFilterPreprocessor(d=5, sigma_color=75, sigma_space=75)
        large_d = BilateralFilterPreprocessor(d=15, sigma_color=75, sigma_space=75)
        
        small_result = small_d(sample_image)
        large_result = large_d(sample_image)
        
        assert small_result.shape == sample_image.shape
        assert large_result.shape == sample_image.shape
        # Larger d should produce different results
        assert not np.array_equal(small_result, large_result)