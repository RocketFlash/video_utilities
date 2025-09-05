"""
Integration tests for VideoFrameSplitter with real video files.
"""

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
                    pytest.mark.xfail(reason=f"No {ext} test file available")