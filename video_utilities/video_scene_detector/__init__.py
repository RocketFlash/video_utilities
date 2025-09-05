import os
import cv2
from tqdm.auto import tqdm
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)
import numpy as np
from dataclasses import dataclass
from .config import VideoSceneDetectorConfig
try:
    from scenedetect import (
        detect, 
        open_video,
        SceneManager,
        ContentDetector
    )
    SCENEDETECT_INSTALLED = True
except ImportError:
    SCENEDETECT_INSTALLED = False

@dataclass
class SceneData:
    """Data class representing a video scene with timing information."""
    scene_id: int
    start_frame: int
    end_frame: int
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    n_frames: Optional[int] = None
    l_sec: Optional[float] = None

    

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        if self.start_frame > self.end_frame:
            raise ValueError(f"start_frame ({self.start_frame}) must be less than end_frame ({self.end_frame})")
        
        # Calculate n_frames if not provided
        if self.n_frames is None:
            self.n_frames = self.end_frame - self.start_frame + 1
        
        # Calculate l_sec if we have timing info but not duration
        if self.l_sec is None and self.start_sec is not None and self.end_sec is not None:
            self.l_sec = self.end_sec - self.start_sec

    def calculate_timing(self, fps: float) -> 'SceneData':
        """Calculate timing information based on FPS."""
        if self.start_sec is None:
            self.start_sec = round(self.start_frame / fps, 3)
        if self.end_sec is None:
            self.end_sec = round(self.end_frame / fps, 3)
        if self.l_sec is None:
            self.l_sec = self.end_sec - self.start_sec
        return self

    def __str__(self) -> str:
        duration_str = f"{self.l_sec:.2f}s" if self.l_sec is not None else f"{self.n_frames or 0} frames"
        return (
            f"Scene {self.scene_id}: frames {self.start_frame}-{self.end_frame} "
            f"({self.n_frames or 0} frames, {duration_str})"
        )

class VideoSceneDetector:
    r"""
    VideoSceneDetector wrapper around PySceneDetect functionality (https://github.com/Breakthrough/PySceneDetect/tree/main)

    Args:
        threshold (`float`, *optional*, defaults to `27`):
            threshold value for scene detector
        min_scene_len (`int`, *optional*, defaults to `15`):
            Once a cut is detected, this many frames must pass before a new one can
            be added to the scene list.
        show_progress (`bool`, *optional*, defaults to `None`):
            Initial frame second. If not None `start_idx` argument will be ignored
    """
    def __init__(
        self,
        config: Optional[VideoSceneDetectorConfig] = None,
        threshold: float = 27.0,
        min_scene_len: int = 15,
        show_progress: bool = True,
        backend: str = 'pyav'
    ):
        if config is None:
            config = VideoSceneDetectorConfig(
                threshold=threshold,
                min_scene_len=min_scene_len,
                show_progress=show_progress,
                backend=backend
            )
        self.config = config
        self.set_params_from_config(config=config)


    def set_params_from_config(
        self, 
        config: VideoSceneDetectorConfig
    ):
        self.threshold = config.threshold
        self.min_scene_len = config.min_scene_len
        self.show_progress = config.show_progress
        self.backend = config.backend
        

    def detect_scenes(
        self, 
        video_path
    ):
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                min_scene_len=self.min_scene_len,
                threshold=self.threshold
            )
        )

        video = open_video(
            video_path,
            backend=self.backend
        )
        
        scene_manager.detect_scenes(
            video, 
            show_progress=self.show_progress
        )
        return scene_manager.get_scene_list()


    def get_scene(
        self, 
        scene_raw,
        scene_id
    ):
        start_frame = scene_raw[0].get_frames()
        end_frame = scene_raw[1].get_frames()
        
        start_sec = scene_raw[0].get_seconds()
        end_sec = scene_raw[1].get_seconds()
    
        n_frames = end_frame - start_frame
        l_sec = end_sec - start_sec
        
        scene = SceneData(
            scene_id=scene_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
            n_frames=n_frames,
            l_sec=l_sec
        )
        return scene


    def get_scene_list(
        self, 
        scenes_raw
    ):
        scene_list = []
        for scene_id, scene_raw in enumerate(scenes_raw):
            scene = self.get_scene(
                scene_raw=scene_raw, 
                scene_id=scene_id
            )
            scene_list.append(scene)
        return scene_list
      
    
    def __call__(
        self,
        video_path: Union[str, os.PathLike],
    ):
        if not SCENEDETECT_INSTALLED:
            print('Scenedetect is not installed. Install it via `pip install scenedetect`')
            return []
        
        scenes_raw = self.detect_scenes(str(video_path))
        scene_list = self.get_scene_list(scenes_raw)
        return scene_list