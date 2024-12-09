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
from scenedetect import (
    detect, 
    open_video,
    SceneManager,
    ContentDetector
)
    

@dataclass
class SceneData:
    scene_id: int
    start_frame: int
    end_frame: int
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    n_frames: Optional[int] = None
    l_sec: Optional[float] = None

    def __str__(self):
        info_str = f"Scene data info:\n"
        info_str += f"scene id   : {self.scene_id}\n"
        info_str += f"start frame: {self.start_frame}\n"
        info_str += f"end frame  : {self.end_frame}\n"
        info_str += f"start sec  : {self.start_sec}\n"
        info_str += f"end sec    : {self.end_sec}\n"
        info_str += f"n frames   : {self.n_frames}\n"
        info_str += f"len in sec : {self.l_sec}\n"
        return info_str


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
        show_progress: bool = True
    ):
        if config is None:
            config = VideoSceneDetectorConfig(
                threshold=threshold,
                min_scene_len=min_scene_len,
                show_progress=show_progress
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
        

    def detect_scenes(self, video_path):
        video = open_video(video_path)

        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                min_scene_len=self.min_scene_len,
                threshold=self.threshold
            )
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
        scenes_raw = self.detect_scenes(str(video_path))
        scene_list = self.get_scene_list(scenes_raw)
        return scene_list