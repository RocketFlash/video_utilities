import pytest
import time
import numpy as np
from video_utilities import VideoFrameSplitter, VideoFrameSplitterConfig

@pytest.mark.slow
class TestVideoFrameSplitterPerformance:
    """Performance tests for VideoFrameSplitter."""
    
    @pytest.mark.parametrize("config_params,expected_max_time", [
        ({"frame_max_size": 128, "n_frames_max": 5}, 2.0),
        ({"frame_max_size": 256, "n_frames_max": 10}, 5.0),
        ({"frame_max_size": 512, "n_frames_max": 20}, 10.0),
    ])
    def test_extraction_performance(self, config_params, expected_max_time, sample_video_path):
        """Test that frame extraction completes within reasonable time."""
        config = VideoFrameSplitterConfig(
            show_progress=False,
            **config_params
        )
        splitter = VideoFrameSplitter(config)
        
        start_time = time.time()
        result = splitter.extract_frames(sample_video_path)
        elapsed = time.time() - start_time
        
        assert result is not None
        assert elapsed < expected_max_time, f"Extraction took {elapsed:.2f}s, expected < {expected_max_time}s"
    
    def test_memory_usage_large_frames(self, sample_video_path):
        """Test memory usage doesn't explode with large frame requests."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = VideoFrameSplitterConfig(
            n_frames_max=50,
            frame_max_size=512,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(sample_video_path)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert result is not None
        # Should not use more than 500MB additional memory
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
    
    @pytest.mark.benchmark
    def test_benchmark_different_sizes(self, benchmark, sample_video_path):
        """Benchmark frame extraction with different frame sizes."""
        config = VideoFrameSplitterConfig(
            frame_max_size=256,
            n_frames_max=10,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        
        def extract_frames():
            return splitter.extract_frames(sample_video_path)
        
        result = benchmark(extract_frames)
        assert result is not None