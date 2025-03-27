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
    video_reader_type (`str`, defaults to `decord`):
        Video reader backend type. Could be one of ['opencv', 'decord']
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
    min_n_frames_per_scene (`int`, defaults to `1`):
        minimum number of selected frames from scene. Applicable if the splitting is based on scenes data
    max_n_frames_per_scene (`int`, defaults to `10`):
        maximum number of selected frames from scene. Applicable if the splitting is based on scenes data
    scene_length_threshold (`float`, defaults to `30`):
        minimum scene length in seconds at which max_n_frames_per_scene will be selected
    """
    video_reader_type: str = 'decord'
    start_idx: int = 0
    frame_interval: int = 1
    start_sec: Optional[float] = None
    frame_interval_sec: Optional[float] = None
    frame_max_size: Optional[int] = 512
    n_frames_max: Optional[int] = None
    n_sec_max: Optional[float] = None
    min_n_frames_per_scene: int = 3
    max_n_frames_per_scene: int = 20
    scene_length_threshold: float = 60