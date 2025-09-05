import pytest
from video_utilities import SceneData

class TestSceneData:
    """Test SceneData functionality."""
    
    def test_scene_data_creation(self):
        """Test SceneData creation and basic properties."""
        scene = SceneData(scene_id=1, start_frame=100, end_frame=200)
        
        assert scene.scene_id == 1
        assert scene.start_frame == 100
        assert scene.end_frame == 200
        assert scene.n_frames == 101  # inclusive range
    
    def test_scene_timing_calculation(self):
        """Test scene timing calculation with different FPS values."""
        scene = SceneData(scene_id=0, start_frame=0, end_frame=299)
        scene.calculate_timing(fps=30.0)
        
        assert scene.start_sec == 0.0
        assert abs(scene.end_sec - 9.967) < 0.01  # 299/30
        assert abs(scene.l_sec - 10.0) < 0.1      # 300 frames / 30 fps
    
    @pytest.mark.parametrize("start_frame,end_frame,expected_frames", [
        (0, 1, 2),      # Adjusted for your validation logic
        (0, 9, 10),
        (10, 19, 10),
        (100, 199, 100),
    ])
    def test_scene_frame_counting(self, start_frame, end_frame, expected_frames):
        """Test that frame counting is correct for various ranges."""
        scene = SceneData(scene_id=0, start_frame=start_frame, end_frame=end_frame)
        assert scene.n_frames == expected_frames
    
    def test_scene_validation(self):
        """Test SceneData validation logic."""
        # Valid scene
        scene = SceneData(scene_id=0, start_frame=10, end_frame=20)
        assert scene.n_frames == 11
        
        # Invalid scene - start >= end
        with pytest.raises(ValueError, match="start_frame.*must be less than.*end_frame"):
            SceneData(scene_id=0, start_frame=20, end_frame=10)