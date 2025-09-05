import pytest
import numpy as np
from abc import ABC
from video_utilities.frame_preprocessor import FramePreprocessor


class DummyPreprocessor(FramePreprocessor):
    """Dummy preprocessor for testing the base class."""
    
    def __init__(self, multiplier=1.0, name="dummy"):
        self.multiplier = multiplier
        self.name = name
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return (frame * self.multiplier).astype(frame.dtype)
    
    def get_name(self) -> str:
        return self.name
    
    def get_config(self):
        return {
            'type': 'DummyPreprocessor',
            'multiplier': self.multiplier,
            'name': self.name
        }


class TestFramePreprocessor:
    """Test the abstract base preprocessor class."""
    
    def test_abstract_class(self):
        """Test that FramePreprocessor is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            FramePreprocessor()
    
    def test_concrete_implementation(self, sample_image):
        """Test that concrete implementation works correctly."""
        preprocessor = DummyPreprocessor(multiplier=2.0, name="test_dummy")
        
        # Test processing
        result = preprocessor(sample_image)
        expected = (sample_image * 2.0).astype(sample_image.dtype)
        np.testing.assert_array_equal(result, expected)
        
        # Test name
        assert preprocessor.get_name() == "test_dummy"
        
        # Test config
        config = preprocessor.get_config()
        assert config['type'] == 'DummyPreprocessor'
        assert config['multiplier'] == 2.0
        assert config['name'] == 'test_dummy'
    
    def test_repr(self):
        """Test string representation."""
        preprocessor = DummyPreprocessor(multiplier=1.5)
        repr_str = repr(preprocessor)
        assert 'DummyPreprocessor' in repr_str
        assert 'multiplier' in repr_str
        assert '1.5' in repr_str
    
    def test_multiple_calls(self, sample_image):
        """Test that preprocessor is stateless and consistent."""
        preprocessor = DummyPreprocessor(multiplier=0.5)
        
        result1 = preprocessor(sample_image)
        result2 = preprocessor(sample_image)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_image_types(self):
        """Test preprocessor with different image types."""
        preprocessor = DummyPreprocessor(multiplier=2.0)
        
        # Test with uint8
        img_uint8 = np.random.randint(0, 128, (100, 100, 3), dtype=np.uint8)
        result_uint8 = preprocessor(img_uint8)
        assert result_uint8.dtype == np.uint8
        
        # Test with float32
        img_float32 = np.random.random((100, 100, 3)).astype(np.float32)
        result_float32 = preprocessor(img_float32)
        assert result_float32.dtype == np.float32
        
        # Test with grayscale
        img_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result_gray = preprocessor(img_gray)
        assert result_gray.shape == img_gray.shape
        assert result_gray.dtype == np.uint8