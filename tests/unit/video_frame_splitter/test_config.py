import pytest
from video_utilities import VideoFrameSplitterConfig, VideoReaderType

class TestVideoFrameSplitterConfig:
    """Test VideoFrameSplitterConfig functionality."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = VideoFrameSplitterConfig()
        
        assert config.start_idx == 0
        assert config.frame_interval == 1
        assert config.n_frames_max is None
        assert config.show_progress is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = VideoFrameSplitterConfig(
            start_idx=0,
            frame_interval=1,
            n_frames_max=10
        )
        assert config.start_idx == 0
        assert config.frame_interval == 1
        
        # Invalid config - negative start_idx
        with pytest.raises(ValueError, match="start_idx must be non-negative"):
            VideoFrameSplitterConfig(start_idx=-1)
        
        # Invalid config - zero frame_interval
        with pytest.raises(ValueError, match="frame_interval must be at least 1"):
            VideoFrameSplitterConfig(frame_interval=0)
        
        # Invalid config - scene parameters
        with pytest.raises(ValueError, match="min_n_frames_per_scene must be at least 1"):
            VideoFrameSplitterConfig(min_n_frames_per_scene=0)
    
    @pytest.mark.parametrize("reader_type", [
        VideoReaderType.OPENCV,
        VideoReaderType.AUTO,
    ])
    def test_reader_type_config(self, reader_type):
        """Test video reader type configuration."""
        config = VideoFrameSplitterConfig(video_reader_type=reader_type)
        assert config.video_reader_type == reader_type
    
    def test_scene_config_parameters(self):
        """Test scene-related configuration parameters."""
        config = VideoFrameSplitterConfig(
            min_n_frames_per_scene=3,
            max_n_frames_per_scene=10,
            scene_length_threshold=60.0
        )
        
        assert config.min_n_frames_per_scene == 3
        assert config.max_n_frames_per_scene == 10
        assert config.scene_length_threshold == 60.0
        
        # Invalid: max < min
        with pytest.raises(ValueError, match="max_n_frames_per_scene must be >= min_n_frames_per_scene"):
            VideoFrameSplitterConfig(
                min_n_frames_per_scene=10,
                max_n_frames_per_scene=5
            )