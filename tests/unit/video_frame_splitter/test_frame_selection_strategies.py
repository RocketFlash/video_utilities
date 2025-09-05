import pytest
from video_utilities import (
    VideoFrameSplitter, 
    VideoFrameSplitterConfig, 
    FrameSelectionStrategy,
    SceneData
)

class TestFrameSelectionStrategies:
    """Test different frame selection strategies."""
    
    @pytest.fixture(autouse=True)
    def setup_video_info(self, sample_video_path):
        """Get video info for test validation."""
        import cv2
        cap = cv2.VideoCapture(str(sample_video_path))
        self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_duration = self.video_frame_count / self.video_fps if self.video_fps > 0 else 0
        cap.release()
    
    @pytest.mark.parametrize("selection_params,expected_strategy", [
        ({"frame_interval": 10, "n_frames_max": 5}, FrameSelectionStrategy.INTERVAL),
        ({"frame_interval_sec": 1.0, "n_sec_max": 5.0}, FrameSelectionStrategy.INTERVAL),
        ({"n_random_frames": 3, "start_sec": 1.0, "n_sec_max": 8.0}, FrameSelectionStrategy.RANDOM),
    ])
    def test_selection_strategies(self, selection_params, expected_strategy, sample_video_path):
        """Test different frame selection strategies."""
        # Validate params against actual video
        if "n_sec_max" in selection_params:
            selection_params["n_sec_max"] = min(selection_params["n_sec_max"], self.video_duration - 1)
        
        if "start_sec" in selection_params:
            selection_params["start_sec"] = min(selection_params["start_sec"], self.video_duration / 2)
        
        config = VideoFrameSplitterConfig(
            show_progress=False,
            **selection_params
        )
        splitter = VideoFrameSplitter(config)
        
        result = splitter.extract_frames(sample_video_path)
        
        assert result is not None, f"Failed with params: {selection_params}"
        assert result.total_frames > 0, f"No frames extracted with params: {selection_params}"
        assert result.selection_strategy == expected_strategy
    
    def test_manual_frame_selection(self, sample_video_path):
        """Test manual frame selection by indices and timestamps."""
        config = VideoFrameSplitterConfig(show_progress=False)
        splitter = VideoFrameSplitter(config)
        
        # Test by frame indices
        max_frame = min(50, self.video_frame_count - 1)
        manual_indices = [0, max_frame // 4, max_frame // 2, max_frame]
        manual_indices = [idx for idx in manual_indices if idx < self.video_frame_count]
        
        result = splitter.extract_frames(
            sample_video_path, 
            selected_frame_idxs=manual_indices
        )
        
        assert result is not None
        assert result.selection_strategy == FrameSelectionStrategy.MANUAL
        assert result.total_frames <= len(manual_indices)
    
    def test_scene_based_selection(self, sample_video_path):
        """Test scene-based frame selection."""
        # Create scenes that fit within the actual video
        scenes_count = 4
        frames_per_scene = self.video_frame_count // scenes_count
        
        scenes = []
        for i in range(scenes_count):
            start_frame = i * frames_per_scene
            end_frame = min((i + 1) * frames_per_scene - 1, self.video_frame_count - 1)
            
            scene = SceneData(scene_id=i, start_frame=start_frame, end_frame=end_frame)
            scene.calculate_timing(self.video_fps)
            scenes.append(scene)
        
        config = VideoFrameSplitterConfig(
            min_n_frames_per_scene=2,
            max_n_frames_per_scene=5,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        
        result = splitter.extract_frames(sample_video_path, scene_list=scenes)
        
        assert result is not None
        assert result.selection_strategy == FrameSelectionStrategy.SCENE_BASED
        assert result.total_frames > 0