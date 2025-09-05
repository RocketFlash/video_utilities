import pytest
import numpy as np
from video_utilities.frame_preprocessor import *

class TestPreprocessorComposer:
    
    def test_composer_creation(self):
        """Test basic composer creation."""
        preprocessors = [
            ResizePreprocessor(320, 240),
            BrightnessContrastPreprocessor(brightness=10)
        ]
        composer = PreprocessorComposer(preprocessors, "test")
        
        assert len(composer) == 2
        assert composer.get_name() == "test"
        assert composer.get_preprocessor_names() == ["resize", "brightness_contrast"]
    
    def test_composer_empty_list(self):
        """Test that empty preprocessor list raises error."""
        with pytest.raises(ValueError):
            PreprocessorComposer([])
    
    def test_composer_invalid_preprocessor(self):
        """Test that invalid preprocessor raises error."""
        with pytest.raises(TypeError):
            PreprocessorComposer([ResizePreprocessor(320, 240), "not_a_preprocessor"])
    
    def test_composer_processing(self, sample_image):
        """Test that composer applies preprocessors in order."""
        original_shape = sample_image.shape
        
        composer = PreprocessorComposer([
            ResizePreprocessor(320, 240),
            BrightnessContrastPreprocessor(brightness=20, contrast=1.2)
        ])
        
        result = composer(sample_image)
        
        # Should be resized
        assert result.shape[:2] == (240, 320)
        # Should be brighter (higher mean)
        assert result.mean() > sample_image.mean()
    
    def test_composer_add_remove(self):
        """Test adding and removing preprocessors."""
        composer = PreprocessorComposer([ResizePreprocessor(320, 240)])
        
        # Add preprocessor
        composer.add_preprocessor(BrightnessContrastPreprocessor())
        assert len(composer) == 2
        
        # Insert preprocessor
        composer.insert_preprocessor(0, CropPreprocessor(10, 10, 310, 230))
        assert len(composer) == 3
        assert composer[0].get_name() == "crop"
        
        # Remove preprocessor
        composer.remove_preprocessor(1)
        assert len(composer) == 2
        assert composer[1].get_name() == "brightness_contrast"
    
    def test_composer_config(self):
        """Test composer configuration serialization."""
        composer = PreprocessorComposer([
            ResizePreprocessor(320, 240),
            BrightnessContrastPreprocessor(brightness=10)
        ], "test_config")
        
        config = composer.get_config()
        
        assert config['type'] == 'PreprocessorComposer'
        assert config['name'] == 'test_config'
        assert config['count'] == 2
        assert len(config['preprocessors']) == 2

class TestConditionalPreprocessor:
    
    def test_conditional_true(self, sample_image):
        """Test conditional preprocessor when condition is True."""
        def always_true(frame):
            return True
        
        conditional = ConditionalPreprocessor(
            BrightnessContrastPreprocessor(brightness=50),
            always_true,
            "test"
        )
        
        result = conditional(sample_image)
        assert result.mean() > sample_image.mean()  # Should be brighter
    
    def test_conditional_false(self, sample_image):
        """Test conditional preprocessor when condition is False."""
        def always_false(frame):
            return False
        
        conditional = ConditionalPreprocessor(
            BrightnessContrastPreprocessor(brightness=50),
            always_false,
            "test"
        )
        
        result = conditional(sample_image)
        np.testing.assert_array_equal(result, sample_image)  # Should be unchanged
    
    def test_conditional_based_on_brightness(self):
        """Test conditional preprocessor based on image brightness."""
        # Create dark image
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8) + 30
        
        # Create bright image  
        bright_image = np.zeros((100, 100, 3), dtype=np.uint8) + 200
        
        def is_dark(frame):
            return frame.mean() < 100
        
        conditional = ConditionalPreprocessor(
            BrightnessContrastPreprocessor(brightness=50),
            is_dark,
            "dark_enhancer"
        )
        
        # Dark image should be enhanced
        dark_result = conditional(dark_image)
        assert dark_result.mean() > dark_image.mean()
        
        # Bright image should remain unchanged
        bright_result = conditional(bright_image)
        np.testing.assert_array_equal(bright_result, bright_image)