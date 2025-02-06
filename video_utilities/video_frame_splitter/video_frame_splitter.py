import os
import cv2
import numpy as np
from tqdm.auto import tqdm
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)
from dataclasses import dataclass
from .config import VideoFrameSplitterConfig
from ..video_scene_detector import SceneData


@dataclass
class VideoFrame:
    image: np.ndarray
    idx: int
    timestamp: float
    scene_id: Optional[int]
    

@dataclass
class VideoFramesData:
    frames: List
    frame_w_orig: int
    frame_h_orig: int
    video_len_orig: float
    n_frames_orig: int
    fps: float

    frame_w: int
    frame_h: int
    start_idx: int
    start_sec: float
    n_frames_max: int
    n_sec_max: float
    frame_interval: int
    frame_interval_sec: float
    video_len: float
    is_scene_based_selection: bool
    selected_n_frames: int
    selected_frame_idxs: List[int]
    selected_frame_scene_ids: List[int]
    
    def __str__(self):
        info_str = f"Original video info:\n"
        info_str += f"    original video frame width     : {self.frame_w_orig}\n"
        info_str += f"    original video frame height    : {self.frame_h_orig}\n"
        info_str += f"    original video len in seconds  : {self.video_len_orig}\n"
        info_str += f"    original video len in n frames : {self.n_frames_orig}\n"
        info_str += f"    fps : {self.fps}\n"
        info_str += f"\n"
        info_str += f"Splitted video info:\n"
        info_str += f"    frame width    : {self.frame_w}\n"
        info_str += f"    frame height   : {self.frame_h}\n"
        info_str += f"    start index    : {self.start_idx}\n"
        info_str += f"    start second   : {self.start_sec}\n"
        info_str += f"    n frames max   : {self.n_frames_max}\n"
        info_str += f"    n seconds max  : {self.n_sec_max}\n"
        info_str += f"    frame interval : {self.frame_interval}\n"
        info_str += f"    frame interval in seconds: {self.frame_interval_sec}\n"
        info_str += f"    video len in seconds     : {self.video_len}\n"
        info_str += f"    scene based selection    : {self.is_scene_based_selection}\n"
        info_str += f"    selected n frames        : {self.selected_n_frames}\n"
        info_str += f"    selected frame indexes   : {self.selected_frame_idxs}\n"
        info_str += f"    selected frame scene ids : {self.selected_frame_scene_ids}\n"
        return info_str
    
    
    def get_scene_frames_dict(self):
        scene_frames_dict = {}
        for idx, scene_id in enumerate(self.selected_frame_scene_ids):
            if scene_id not in scene_frames_dict:
                scene_frames_dict[scene_id] = []
            scene_frames_dict[scene_id].append(self.frames[idx])
        return scene_frames_dict


class VideoFrameSplitter:
    def __init__(
        self,
        config: Optional[VideoFrameSplitterConfig] = None,
    ):  
        if config is None:
            config = self.get_default_config()

        self.config = config
        self.set_params_from_config(config=config)


    def get_default_config(self):
        return VideoFrameSplitterConfig()


    def set_params_from_config(
        self, 
        config: VideoFrameSplitterConfig
    ):
        for key, value in vars(config).items():
            setattr(self, key, value)


    def get_frame_idxs_from_interval(
        self,
        start_idx: int, 
        end_idx: int, 
        n_frames: int = 1
    ):
        frame_indexes_list = list(range(start_idx, end_idx))
        if n_frames >= len(frame_indexes_list):
            return frame_indexes_list
        else:
            if n_frames==1:
                return [int((end_idx + start_idx)/2)]
                
            indices = np.linspace(0, len(frame_indexes_list) - 1, n_frames, dtype=int)
            return [frame_indexes_list[i] for i in indices]
    

    def get_selected_frames_from_scenes(
        self,
        scene_list: List[SceneData],
        start_idx: int,
        n_frames_total: int,
        n_frames_per_scene: int,
    ):
        if not scene_list:
            scene_list = [
                SceneData(
                    scene_id=0,
                    start_frame=0,
                    end_frame=n_frames_total,
                )
            ]

        selected_frame_idxs = []
        selected_frame_scene_ids = []

        for scene in scene_list:
            start_scene_idx = scene.start_frame
            end_scene_idx = scene.end_frame

            if start_idx>=end_scene_idx:
                continue
            elif start_idx<end_scene_idx and start_idx>start_scene_idx:
                start_scene_idx = start_idx

            scene_selected_frame_idxs = self.get_frame_idxs_from_interval(
                start_scene_idx, 
                end_scene_idx, 
                n_frames=n_frames_per_scene
            )

            selected_frame_idxs += scene_selected_frame_idxs
            selected_frame_scene_ids += [scene.scene_id] * len(scene_selected_frame_idxs)

        return selected_frame_idxs, selected_frame_scene_ids
      
    
    def __call__(
        self,
        video_path: Union[str, os.PathLike],
        scene_list: Optional[List[SceneData]] = None,
        n_frames_per_scene: Optional[int] = 1,
    ):
        frames = []

        video = cv2.VideoCapture(str(video_path))
        
        if not video.isOpened(): 
            return None
            
        n_frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w_orig = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h_orig = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        video_len_orig = n_frames_total / fps if fps>0 else -1

        if self.frame_interval_sec is not None:
            frame_interval = int(self.frame_interval_sec * fps)
        else:
            frame_interval = self.frame_interval
            
        if self.start_sec is not None:
            start_idx = int(self.start_sec * fps)
        else:
            start_idx = self.start_idx
    
        if start_idx>=n_frames_total:
            print(f'Can not start from {self.start_idx} frame, because total number of frames is {n_frames_total}')
            start_idx = 0

        frame_interval_sec = frame_interval / fps if fps>0 else -1
        start_sec = start_idx / fps if fps>0 else -1

        n_frames_available = (n_frames_total - start_idx) // frame_interval
        n_frames_max = self.n_frames_max
        n_frames = n_frames_available
        n_sec_max = self.n_sec_max

        if n_sec_max is not None:
            n_frames_max = int(n_sec_max * fps / frame_interval)

        if n_frames_max is not None:
            if n_frames_available < n_frames_max:
                print(f'Can not read {n_frames_max} frames because only {n_frames_available} frames are available')
            n_frames = min(n_frames_max, n_frames_available)

        video.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        selected_frame_idxs = []
        selected_frame_scene_ids = []
        is_scene_based_selection = False
        
        if scene_list is not None:
            is_scene_based_selection = True
        
            (selected_frame_idxs, 
             selected_frame_scene_ids) = self.get_selected_frames_from_scenes(
                scene_list=scene_list,
                start_idx=start_idx,
                n_frames_total=n_frames_total,
                n_frames_per_scene=n_frames_per_scene,
            )

            n_frames = len(selected_frame_idxs)
            frame_interval = None
            frame_interval_sec = None
            n_sec_max = None
            n_frames_max = None

        resize_scale = 1
        if self.frame_max_size is not None:
            max_dim = max(frame_h_orig, frame_w_orig)
            if max_dim > self.frame_max_size:
                resize_scale = self.frame_max_size / max_dim

        frame_idx = 0
        frame_count = 0

        pbar = tqdm(total=n_frames)
        while True:
            if frame_count>=n_frames:
                break
                
            success = video.grab()
            if not success:
                break

            if frame_idx < start_idx:
                frame_idx += 1
                continue
            
            is_selected_frame = False
            scene_id = None

            if is_scene_based_selection:
                if frame_idx in selected_frame_idxs :
                    is_selected_frame = True
                    scene_list_idx = selected_frame_idxs.index(frame_idx)
                    scene_id = selected_frame_scene_ids[scene_list_idx]
            else:
                is_selected_frame = ((start_idx + frame_idx) % frame_interval) == 0
                if is_selected_frame:
                    selected_frame_idxs.append(frame_idx)
            
            if is_selected_frame:
                success, image = video.retrieve()
                if not success:
                    break
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                timestamp = round(frame_idx / fps, 3)

                if resize_scale != 1:
                    image = cv2.resize(
                        image, 
                        (0, 0),
                        fx=resize_scale, 
                        fy=resize_scale
                    )

                frame_data = VideoFrame(
                    image=image,
                    idx=frame_idx,
                    timestamp=timestamp,
                    scene_id=scene_id
                )

                frames.append(frame_data)
                frame_count += 1
                pbar.update(1)
            frame_idx += 1

        video.release()
        pbar.close()

        frame_h, frame_w = image.shape[:2]
        
        if fps>0:
            video_start_sec = selected_frame_idxs[0] / fps
            video_end_sec = selected_frame_idxs[-1] / fps
            video_len = round(video_end_sec - video_start_sec, 3)
        else:
            video_len = -1
        
        video_frames_data = VideoFramesData(
            frames=frames,
            frame_w_orig=frame_w_orig,
            frame_h_orig=frame_h_orig,
            video_len_orig=video_len_orig,
            n_frames_orig=n_frames_total,
            fps=fps,
            frame_w=frame_w,
            frame_h=frame_h,
            start_idx=start_idx,
            start_sec=start_sec,
            n_frames_max=n_frames_max,
            n_sec_max=n_sec_max,
            frame_interval=frame_interval,
            frame_interval_sec=frame_interval_sec,
            video_len=video_len,
            is_scene_based_selection=is_scene_based_selection,
            selected_n_frames=len(frames),
            selected_frame_idxs=selected_frame_idxs,
            selected_frame_scene_ids=selected_frame_scene_ids,            
        )

        return video_frames_data