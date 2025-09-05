# """
# Integration tests for full processing pipeline.
# """

# import pytest
# import numpy as np
# from pathlib import Path
# import tempfile

# from pose_video_processor import PoseVideoProcessor, VideoProcessingConfig
# from video_utilities.frame_preprocessor import *

# class TestFullPipeline:
    
#     def test_processor_with_preprocessors(self, sample_image_path, temp_output_dir):
#         """Test video processor with preprocessing pipeline."""
#         # Create preprocessing pipeline
#         preprocessing_pipeline = PreprocessorComposer([
#             ResizePreprocessor(320, 240),
#             BrightnessContrastPreprocessor(brightness=10, contrast=1.1),
#             GaussianBlurPreprocessor(kernel_size=3, sigma=0.5)
#         ], "test_pipeline")
        
#         # Create processing config
#         config = VideoProcessingConfig(
#             frame_preprocessors=[preprocessing_pipeline],
#             filter_window_size=3
#         )
        
#         # This would require your actual video processor implementation
#         # processor = PoseVideoProcessor(processing_config=config)
        
#         # For now, just test that config is created correctly
#         assert len(config.frame_preprocessors) == 1
#         assert config.frame_preprocessors[0].get_name() == "test_pipeline"
#         assert len(config.frame_preprocessors[0]) == 3
    
#     def test_complex_preprocessing_chain(self, sample_image):
#         """Test complex preprocessing chain."""
#         # Create complex pipeline
#         pipeline = PreprocessorComposer([
#             CropPreprocessor(50, 50, 590, 430),
#             ResizePreprocessor(512, 384),
#             CLAHEPreprocessor(clip_limit=2.0),
#             BrightnessContrastPreprocessor(brightness=10, contrast=1.1),
#             ConditionalPreprocessor(
#                 GaussianBlurPreprocessor(kernel_size=3),
#                 lambda frame: frame.mean() > 100,
#                 "conditional_blur"
#             )
#         ], "complex_pipeline")
        
#         result = pipeline(sample_image)
        
#         # Check final dimensions
#         assert result.shape[:2] == (384, 512)
        
#         # Check that processing was applied (image should be different)
#         assert not np.array_equal(result, sample_image)