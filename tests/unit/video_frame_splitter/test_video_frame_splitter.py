import pytest
import numpy as np
import time
from pathlib import Path
import cv2

from video_utilities import (
    VideoFrameSplitter, 
    VideoFrameSplitterConfig, 
    VideoReaderType, 
    FrameSelectionStrategy
)

class TestVideoFrameSplitter:
    """Test core VideoFrameSplitter functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_video_info(self, sample_video_path):
        """Get video info for test validation."""
        cap = cv2.VideoCapture(str(sample_video_path))
        self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_duration = self.video_frame_count / self.video_fps if self.video_fps > 0 else 0
        cap.release()
    
    def test_splitter_initialization(self):
        """Test VideoFrameSplitter initialization."""
        config = VideoFrameSplitterConfig()
        splitter = VideoFrameSplitter(config)
        assert splitter.config == config
    
    def test_basic_frame_extraction(self, sample_video_path):
        """Test basic frame extraction functionality."""
        config = VideoFrameSplitterConfig(
            n_frames_max=5,
            frame_max_size=224,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        
        result = splitter.extract_frames(sample_video_path)
        
        assert result is not None
        assert result.total_frames <= 5
        assert result.frames is not None
        assert len(result.frames) == result.total_frames
        
        if result.frames:
            first_frame = result.frames[0]
            assert hasattr(first_frame, 'image')
            assert hasattr(first_frame, 'idx')
            assert hasattr(first_frame, 'timestamp')
            assert first_frame.image.shape[0] <= 224
            assert first_frame.image.shape[1] <= 224
    
    def test_invalid_video_path(self):
        """Test handling of invalid video path."""
        config = VideoFrameSplitterConfig(show_progress=False)
        splitter = VideoFrameSplitter(config)
        
        result = splitter.extract_frames("nonexistent_video.mp4")
        assert result is None
    
    @pytest.mark.parametrize("reader_type", [
        VideoReaderType.OPENCV,
        VideoReaderType.AUTO,
    ])
    def test_different_readers(self, reader_type, sample_video_path):
        """Test different video reader types."""
        config = VideoFrameSplitterConfig(
            video_reader_type=reader_type,
            n_frames_max=3,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        
        result = splitter.extract_frames(sample_video_path)
        
        if reader_type in [VideoReaderType.AUTO, VideoReaderType.OPENCV]:
            assert result is not None
            assert result.total_frames <= 3