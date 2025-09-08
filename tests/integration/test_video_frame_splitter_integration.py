import pytest
from pathlib import Path
from video_utilities import VideoFrameSplitter, VideoFrameSplitterConfig

@pytest.mark.integration
class TestVideoFrameSplitterIntegration:
    """Integration tests with real video files."""
    
    def test_full_workflow_with_real_video(self, sample_video_path):
        """Test complete workflow with a real video file."""
        # Step 1: Basic extraction
        config = VideoFrameSplitterConfig(
            n_frames_max=10,
            frame_max_size=256,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(sample_video_path)
        
        assert result is not None
        assert result.total_frames <= 10
        assert result.video_path == sample_video_path
        assert result.end_idx is not None
        assert result.end_sec is not None
        
        # Step 2: Extract specific timestamps based on first result
        if result.frames:
            # Use timestamps from first extraction
            timestamps = [frame.timestamp for frame in result.frames[:3]]
            
            specific_result = splitter.extract_frames(
                sample_video_path,
                selected_seconds=timestamps
            )
            
            assert specific_result is not None
            assert specific_result.total_frames <= 3
            
            # Timestamps should be approximately the same
            for i, frame in enumerate(specific_result.frames):
                expected_time = timestamps[i]
                assert abs(frame.timestamp - expected_time) < 1.0  # Within 1 second
    
    def test_video_format_compatibility(self, test_data_dir):
        """Test compatibility with different video formats."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        config = VideoFrameSplitterConfig(
            n_frames_max=3,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        
        for ext in video_extensions:
            video_files = list(test_data_dir.glob(f'*{ext}'))
            
            for video_file in video_files:
                result = splitter.extract_frames(video_file)
                # Should either work or fail gracefully
                if result is not None:
                    assert result.total_frames >= 0
                    assert result.video_path == video_file
                    pytest.mark.xfail(reason=f"No {ext} test file available")
    
    def test_end_time_constraints(self, sample_video_path):
        """Test end_idx and end_sec constraint functionality."""
        # Test with end_sec constraint
        config = VideoFrameSplitterConfig(
            start_sec=1.0,
            end_sec=5.0,
            frame_interval_sec=0.5,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(sample_video_path)
        
        assert result is not None
        assert result.end_sec == 5.0
        assert all(frame.timestamp <= 5.0 for frame in result.frames)
        assert all(frame.timestamp >= 1.0 for frame in result.frames)
        
        # Test with end_idx constraint
        config_idx = VideoFrameSplitterConfig(
            start_idx=30,
            end_idx=60,
            frame_interval=5,
            show_progress=False
        )
        splitter_idx = VideoFrameSplitter(config_idx)
        result_idx = splitter_idx.extract_frames(sample_video_path)
        
        if result_idx is not None:
            assert result_idx.end_idx == 60
            assert all(frame.idx <= 60 for frame in result_idx.frames)
            assert all(frame.idx >= 30 for frame in result_idx.frames)
    
    def test_video_path_preservation(self, sample_video_path):
        """Test that video_path is correctly stored in VideoFramesData."""
        config = VideoFrameSplitterConfig(
            n_frames_max=5,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(sample_video_path)
        
        assert result is not None
        assert result.video_path == sample_video_path
        assert Path(result.video_path).name == Path(sample_video_path).name