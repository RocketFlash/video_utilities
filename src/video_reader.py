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


@dataclass
class VideoFramesData:
    frames: List
    frame_indexes: List
    w_orig: int
    h_orig: int
    w: int
    h: int
    n_frames: int
    l: float
    l_orig: float
    fps: float
    frame_interval: int
    frame_interval_sec: float
    start_idx: int
    start_sec: float

    def __str__(self):
        info_str = f"Video frames data info:\n"
        info_str += f"width          : {self.w}\n"
        info_str += f"height         : {self.h}\n"
        info_str += f"width original : {self.w_orig}\n"
        info_str += f"height original: {self.h_orig}\n"
        info_str += f"start index    : {self.start_idx}\n"
        info_str += f"start second   : {self.start_sec}\n"
        info_str += f"n of frames    : {self.n_frames}\n"
        info_str += f"fps            : {self.fps}\n"
        info_str += f"frame interval : {self.frame_interval}\n"
        info_str += f"frame interval in seconds: {self.frame_interval_sec}\n"
        info_str += f"clip length in seconds   : {self.l}\n"
        info_str += f"original video length in seconds: {self.l_orig}\n"
        return info_str
    

class VideoReader:
    r"""
    VideoReader reads frames and return video information

    Args:
        start_idx (`int`, *optional*, defaults to `0`):
            Initial frame index
        start_sec (`float`, *optional*, defaults to `None`):
            Initial frame second. If not None `start_idx` argument will be ignored
        frame_interval (`int`, *optional*, defaults to `1`):
            Interval between collected frames
        frame_interval_sec (`float`, *optional*, defaults to `None`):
            Interval between frames in seconds. If not None `frame_interval` argument will be ignored
        frame_max_size (`int`, *optional*, defaults to `None`):
            If not None, images whose long side is greater than `frame_max_size` value will be resized 
            while maintaining the aspect ratio.
        n_frames_max (`int`, *optional*, defaults to `None`):
            If not None, the number of frames collected will be limited by this value.
        n_sec_max (`float`, *optional*, defaults to `None`):
            If not None, the number of frames collected will be limited by this value in seconds.
    """
    def __init__(
        self,
        start_idx: int = 0,
        start_sec: Optional[float] = None,
        frame_interval: int = 1,
        frame_interval_sec: Optional[float] = None,
        frame_max_size: Optional[int] = None,
        n_frames_max: Optional[int] = None,
        n_sec_max: Optional[float] = None,
    ):
        self.start_idx = start_idx
        self.start_sec = start_sec
        self.frame_interval = frame_interval
        self.frame_interval_sec = frame_interval_sec
        self.n_frames_max = n_frames_max
        self.n_sec_max = n_sec_max
        self.frame_max_size = frame_max_size

    
    def __call__(
        self,
        video_path: Union[str, os.PathLike],
    ):
        frames = []
        frame_indexes = []

        video = cv2.VideoCapture(str(video_path))
        
        if not video.isOpened(): 
            return None
            
        n_frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        w_orig = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(video.get(cv2.CAP_PROP_FPS))

        l_orig = n_frames_total / fps if fps>0 else -1

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

        if self.n_sec_max is not None:
            n_frames_max = int(self.n_sec_max * fps / frame_interval)

        if n_frames_max is not None:
            if n_frames_available < n_frames_max:
                print(f'Can not read {n_frames_max} frames because only {n_frames_available} frames are available')
            n_frames = min(n_frames_max, n_frames_available)

        video.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        resize_scale = 1
        if self.frame_max_size is not None:
            max_dim = max(h_orig, w_orig)
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
                
            if frame_idx % frame_interval == 0:
                success, frame = video.retrieve()
                if not success:
                    break
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if resize_scale != 1:
                    frame = cv2.resize(
                        frame, 
                        (0, 0),
                        fx=resize_scale, 
                        fy=resize_scale
                    )
                
                frame_indexes.append(frame_idx)
                frames.append(frame)
                frame_count += 1
                pbar.update(1)
            frame_idx += 1

        video.release()
        pbar.close()

        h, w = frame.shape[:2]
        l = len(frames) / fps if fps>0 else -1

        video_frames_data = VideoFramesData(
            frames=frames,
            frame_indexes=frame_indexes,
            w_orig=w_orig,
            h_orig=h_orig,
            w=w,
            h=h,
            n_frames=len(frames),
            l=l,
            l_orig=l_orig,
            fps=fps,
            start_idx=start_idx,
            start_sec=start_sec,
            frame_interval=frame_interval,
            frame_interval_sec=frame_interval_sec
        )

        return video_frames_data