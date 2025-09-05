import pytest
import numpy as np
from video_utilities.frame_preprocessor import (
    FunctionPreprocessor,
    LambdaPreprocessor,
    NoOpPreprocessor
)


def simple_brightness_function(frame):
    """Simple function to increase brightness."""
    return np.clip(frame + 50, 0, 255).astype(frame.dtype)


def invert_function(frame):
    """Function to invert image colors.""" 
    return 255 - frame


def crop_function(frame):
    """Function to crop center 50% of image."""
    h, w = frame.shape[:2]
    start_h, start_w = h // 4, w // 4
    end_h, end_w = start_h + h // 2, start_w + w // 2
    return frame[start_h:end_h, start_w:end_w]


class TestFunctionPreprocessor:
    
    def test_function_preprocessor_basic(self, sample_image):
        """Test basic function preprocessor functionality."""
        preprocessor = FunctionPreprocessor(simple_brightness_function, "brightness_func")
        result = preprocessor(sample_image)
        
        # Should be brighter
        assert result.mean() > sample_image.mean()
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_function_preprocessor_with_kwargs(self, sample_image):
        """Test function preprocessor with additional kwargs."""
        preprocessor = FunctionPreprocessor(
            simple_brightness_function, 
            "brightness_func",
            brightness_value=50,
            author="test"
        )
        result = preprocessor(sample_image)
        
        config = preprocessor.get_config()
        assert config['brightness_value'] == 50
        assert config['author'] == "test"
    
    def test_function_preprocessor_invert(self, sample_image):
        """Test function preprocessor with invert function."""
        preprocessor = FunctionPreprocessor(invert_function, "invert")
        result = preprocessor(sample_image)
        
        # Inverted image + original should be close to 255
        combined = result.astype(np.int32) + sample_image.astype(np.int32)
        expected_sum = np.full_like(combined, 255)
        
        np.testing.assert_allclose(combined, expected_sum, atol=1)
    
    def test_function_preprocessor_crop(self, sample_image):
        """Test function preprocessor with cropping function."""
        preprocessor = FunctionPreprocessor(crop_function, "crop_center")
        result = preprocessor(sample_image)
        
        # Should be half the size in each dimension
        expected_h, expected_w = sample_image.shape[0] // 2, sample_image.shape[1] // 2
        assert result.shape[:2] == (expected_h, expected_w)
    
    def test_function_preprocessor_lambda_inline(self, sample_image):
        """Test function preprocessor with lambda function."""
        # Using lambda to multiply by 0.5
        preprocessor = FunctionPreprocessor(
            lambda frame: (frame * 0.5).astype(frame.dtype),
            "half_brightness"
        )
        result = preprocessor(sample_image)
        
        # Should be darker
        assert result.mean() < sample_image.mean()
    
    def test_function_preprocessor_identity(self, sample_image):
        """Test function preprocessor with identity function."""
        identity_func = lambda x: x
        preprocessor = FunctionPreprocessor(identity_func, "identity")
        result = preprocessor(sample_image)
        
        np.testing.assert_array_equal(result, sample_image)
    
    def test_function_preprocessor_complex_function(self, sample_image):
        """Test function preprocessor with complex function."""
        def complex_function(frame):
            # Apply multiple operations
            # 1. Increase brightness
            bright = np.clip(frame + 30, 0, 255)
            # 2. Increase contrast
            contrasted = np.clip((bright - 128) * 1.2 + 128, 0, 255)
            return contrasted.astype(frame.dtype)
        
        preprocessor = FunctionPreprocessor(complex_function, "complex")
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
        # Should be different from original
        assert not np.array_equal(result, sample_image)
    
    def test_function_preprocessor_error_handling(self):
        """Test function preprocessor error handling."""
        def bad_function(frame):
            raise RuntimeError("Intentional error")
        
        preprocessor = FunctionPreprocessor(bad_function, "bad_func")
        
        # Should propagate the error
        with pytest.raises(RuntimeError):
            preprocessor(np.zeros((100, 100, 3), dtype=np.uint8))
    
    def test_config(self):
        """Test configuration."""
        preprocessor = FunctionPreprocessor(
            simple_brightness_function, 
            "test_func",
            param1="value1",
            param2=42
        )
        config = preprocessor.get_config()
        
        assert config['type'] == 'FunctionPreprocessor'
        assert config['name'] == 'test_func'
        assert config['param1'] == "value1"
        assert config['param2'] == 42
        assert 'function' in config
        assert preprocessor.get_name() == 'test_func'


class TestLambdaPreprocessor:
    
    def test_lambda_preprocessor_basic(self, sample_image):
        """Test basic lambda preprocessor functionality."""
        # Simple brightness adjustment
        preprocessor = LambdaPreprocessor(
            lambda frame: np.clip(frame + 40, 0, 255).astype(frame.dtype),
            "lambda_bright"
        )
        result = preprocessor(sample_image)
        
        assert result.mean() > sample_image.mean()
        assert result.shape == sample_image.shape
    
    def test_lambda_preprocessor_scale(self, sample_image):
        """Test lambda preprocessor with scaling."""
        preprocessor = LambdaPreprocessor(
            lambda frame: (frame * 0.7).astype(frame.dtype),
            "scale_down"
        )
        result = preprocessor(sample_image)
        
        assert result.mean() < sample_image.mean()
        assert result.shape == sample_image.shape
    
    def test_lambda_preprocessor_channel_manipulation(self, sample_image):
        """Test lambda preprocessor with channel manipulation."""
        # Swap red and blue channels
        preprocessor = LambdaPreprocessor(
            lambda frame: frame[:, :, [2, 1, 0]] if len(frame.shape) == 3 else frame,
            "swap_rb"
        )
        result = preprocessor(sample_image)
        
        if len(sample_image.shape) == 3:
            # Red channel of result should equal blue channel of original
            np.testing.assert_array_equal(result[:, :, 0], sample_image[:, :, 2])
            np.testing.assert_array_equal(result[:, :, 2], sample_image[:, :, 0])
            # Green channel should remain the same
            np.testing.assert_array_equal(result[:, :, 1], sample_image[:, :, 1])
    
    def test_lambda_preprocessor_threshold(self, sample_image):
        """Test lambda preprocessor with thresholding."""
        preprocessor = LambdaPreprocessor(
            lambda frame: ((frame > 128) * 255).astype(frame.dtype),
            "threshold"
        )
        result = preprocessor(sample_image)
        
        # Result should only contain 0 and 255 values
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)
    
    def test_lambda_preprocessor_grayscale_conversion(self, sample_image):
        """Test lambda preprocessor for manual grayscale conversion."""
        if len(sample_image.shape) == 3:
            preprocessor = LambdaPreprocessor(
                lambda frame: (0.299 * frame[:, :, 0] + 
                              0.587 * frame[:, :, 1] + 
                              0.114 * frame[:, :, 2]).astype(frame.dtype),
                "manual_gray"
            )
            result = preprocessor(sample_image)
            
            # Should be 2D (grayscale)
            assert len(result.shape) == 2
            assert result.shape[:2] == sample_image.shape[:2]
    
    def test_lambda_preprocessor_mathematical_operations(self, sample_image):
        """Test lambda preprocessor with mathematical operations."""
        # Square root transformation
        preprocessor = LambdaPreprocessor(
            lambda frame: np.sqrt(frame.astype(np.float32) / 255.0).astype(np.float32) * 255.0,
            "sqrt_transform"
        )
        result = preprocessor(sample_image)
        
        assert result.shape == sample_image.shape
        # Square root should brighten dark areas more than bright areas
        assert result.mean() != sample_image.mean()
    
    def test_lambda_preprocessor_identity(self, sample_image):
        """Test lambda preprocessor with identity function."""
        preprocessor = LambdaPreprocessor(lambda frame: frame, "identity")
        result = preprocessor(sample_image)
        
        np.testing.assert_array_equal(result, sample_image)
    
    def test_lambda_preprocessor_error_handling(self):
        """Test lambda preprocessor error handling."""
        def bad_function(frame):
            raise ValueError("Intentional test error")
        
        preprocessor = LambdaPreprocessor(bad_function, "bad_lambda")
        
        with pytest.raises(ValueError):
            preprocessor(np.ones((10, 10, 3), dtype=np.uint8))
    
    def test_config(self):
        """Test configuration."""
        lambda_func = lambda x: x * 2
        preprocessor = LambdaPreprocessor(lambda_func, "test_lambda")
        config = preprocessor.get_config()
        
        assert config['type'] == 'LambdaPreprocessor'
        assert config['name'] == 'test_lambda'
        assert 'function' in config
        assert preprocessor.get_name() == 'test_lambda'


class TestNoOpPreprocessor:
    
    def test_noop_basic(self, sample_image):
        """Test basic NoOp preprocessor functionality."""
        preprocessor = NoOpPreprocessor()
        result = preprocessor(sample_image)
        
        # Should be identical to input
        np.testing.assert_array_equal(result, sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
        assert result is sample_image  # Should be the same object
    
    def test_noop_different_image_types(self):
        """Test NoOp preprocessor with different image types."""
        # Test with uint8
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        preprocessor = NoOpPreprocessor()
        result_uint8 = preprocessor(img_uint8)
        np.testing.assert_array_equal(result_uint8, img_uint8)
        
        # Test with float32
        img_float32 = np.random.random((50, 50, 3)).astype(np.float32)
        result_float32 = preprocessor(img_float32)
        np.testing.assert_array_equal(result_float32, img_float32)
        
        # Test with grayscale
        img_gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result_gray = preprocessor(img_gray)
        np.testing.assert_array_equal(result_gray, img_gray)
    
    def test_noop_custom_name(self, sample_image):
        """Test NoOp preprocessor with custom name."""
        preprocessor = NoOpPreprocessor(name="passthrough")
        result = preprocessor(sample_image)
        
        np.testing.assert_array_equal(result, sample_image)
        assert preprocessor.get_name() == "passthrough"
    
    def test_noop_multiple_calls(self, sample_image):
        """Test NoOp preprocessor consistency across multiple calls."""
        preprocessor = NoOpPreprocessor()
        
        result1 = preprocessor(sample_image)
        result2 = preprocessor(sample_image)
        result3 = preprocessor(result1)
        
        np.testing.assert_array_equal(result1, sample_image)
        np.testing.assert_array_equal(result2, sample_image)
        np.testing.assert_array_equal(result3, sample_image)
    
    def test_noop_with_modified_array(self):
        """Test NoOp preprocessor doesn't create defensive copies."""
        original = np.ones((10, 10, 3), dtype=np.uint8) * 100
        preprocessor = NoOpPreprocessor()
        result = preprocessor(original)
        
        # Modify original
        original[0, 0, 0] = 200
        
        # Result should also be modified (same object)
        assert result[0, 0, 0] == 200
    
    def test_noop_empty_array(self):
        """Test NoOp preprocessor with empty array."""
        empty_array = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        preprocessor = NoOpPreprocessor()
        result = preprocessor(empty_array)
        
        np.testing.assert_array_equal(result, empty_array)
        assert result.shape == empty_array.shape
    
    def test_noop_single_pixel(self):
        """Test NoOp preprocessor with single pixel image."""
        single_pixel = np.array([[[255, 128, 64]]], dtype=np.uint8)
        preprocessor = NoOpPreprocessor()
        result = preprocessor(single_pixel)
        
        np.testing.assert_array_equal(result, single_pixel)
    
    def test_config(self):
        """Test configuration."""
        preprocessor_default = NoOpPreprocessor()
        preprocessor_custom = NoOpPreprocessor(name="custom_noop")
        
        config_default = preprocessor_default.get_config()
        config_custom = preprocessor_custom.get_config()
        
        assert config_default['type'] == 'NoOpPreprocessor'
        assert config_default['name'] == 'noop'
        assert config_custom['name'] == 'custom_noop'
        
        assert preprocessor_default.get_name() == 'noop'
        assert preprocessor_custom.get_name() == 'custom_noop'