from dataclasses import dataclass, asdict, field
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)

@dataclass
class VideoFrameSplitterConfig():
    r"""
    start_idx (`int`, *optional*, defaults to `0`):
        Initial frame index
    frame_interval (`int`, *optional*, defaults to `1`):
        Interval between collected frames
    start_sec (`float`, *optional*, defaults to `None`):
        Initial frame second. If not None `start_idx` argument will be ignored
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
    start_idx: int = 0
    frame_interval: int = 1
    start_sec: Optional[float] = None
    frame_interval_sec: Optional[float] = None
    frame_max_size: Optional[int] = 512
    n_frames_max: Optional[int] = None
    n_sec_max: Optional[float] = None