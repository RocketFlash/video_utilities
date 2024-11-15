import os
import cv2
from tqdm.auto import tqdm
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)
from dataclasses import dataclass
import numpy as np
from scenedetect import (
    detect, 
    open_video,
    SceneManager,
    ContentDetector
)
    

@dataclass
class SceneData:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    n_frames: int
    l_sec: float


    def __str__(self):
        info_str = f"Scene data info:\n"
        info_str += f"start frame: {self.start_frame}\n"
        info_str += f"end frame  : {self.end_frame}\n"
        info_str += f"start sec  : {self.start_sec}\n"
        info_str += f"end sec    : {self.end_sec}\n"
        info_str += f"n frames   : {self.n_frames}\n"
        info_str += f"len in sec : {self.l}\n"
        return info_str
    


class VideoSceneDetector:
    r"""
    VideoSceneDetector wrapper around PySceneDetect functionality (https://github.com/Breakthrough/PySceneDetect/tree/main)

    Args:
        threshold (`float`, *optional*, defaults to `27`):
            threshold value for scene detector
        show_progress (`bool`, *optional*, defaults to `None`):
            Initial frame second. If not None `start_idx` argument will be ignored
    """
    def __init__(
        self,
        threshold: float = 27.0,
        show_progress: bool = True
    ):
        self.threshold = threshold
        self.show_progress = show_progress
        

    def detect_scenes(self, video_path):
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=self.threshold)
        )
        scene_manager.detect_scenes(
            video, 
            show_progress=self.show_progress
        )
        return scene_manager.get_scene_list()


    def get_scene(self, scene_raw):
        start_frame = scene_raw[0].get_frames()
        end_frame = scene_raw[1].get_frames()
        
        start_sec = scene_raw[0].get_seconds()
        end_sec = scene_raw[1].get_seconds()
    
        n_frames = end_frame - start_frame
        l_sec = end_sec - start_sec
        
        scene = SceneData(
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
        for scene_raw in scenes_raw:
            scene = self.get_scene(scene_raw)
            scene_list.append(scene)
        return scene_list
      
    
    def __call__(
        self,
        video_path: Union[str, os.PathLike],
    ):
        scenes_raw = self.detect_scenes(str(video_path))
        scene_list = self.get_scene_list(scenes_raw)
        return scene_list