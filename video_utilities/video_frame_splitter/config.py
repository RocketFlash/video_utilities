from dataclasses import dataclass, asdict, field
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)

@dataclass
class VideoFrameSplitterConfig():
    """
    Configuration for the VideoFrameSplitter.

    video_reader_type (`str`, defaults to `decord`):
        Video reader backend. 'decord' is highly recommended for performance. Falls back to 'opencv'.
    start_idx (`int`, *optional*, defaults to `0`):
        Initial frame index to start selection from.
    start_sec (`float`, *optional*, defaults to `None`):
        Initial second to start selection from. Overrides `start_idx`.
    frame_interval (`int`, *optional*, defaults to `1`):
        Interval between frames for sequential selection.
    frame_interval_sec (`float`, *optional*, defaults to `None`):
        Interval in seconds between frames. Overrides `frame_interval`.
    frame_max_size (`int`, *optional*, defaults to `None`):
        Resizes frames' longer side to this value while maintaining aspect ratio.
    n_frames_max (`int`, *optional*, defaults to `None`):
        Maximum number of frames to extract in a sequence.
    n_sec_max (`float`, *optional*, defaults to `None`):
        Maximum duration in seconds to extract frames from. Overrides `n_frames_max`.
    n_random_frames (`int`, *optional*, defaults to `None`):
        If set, extracts this many random frames from the specified video interval, ignoring other interval logic.
    min_n_frames_per_scene (`int`, defaults to `3`):
        Minimum frames to select from a scene (for scene-based selection).
    max_n_frames_per_scene (`int`, defaults to `20`):
        Maximum frames to select from a scene (for scene-based selection).
    scene_length_threshold (`float`, defaults to `60`):
        Scene duration in seconds to reach `max_n_frames_per_scene`.
    """
    video_reader_type: str = 'decord'
    start_idx: int = 0
    frame_interval: int = 1
    start_sec: Optional[float] = None
    frame_interval_sec: Optional[float] = None
    frame_max_size: Optional[int] = 512
    n_frames_max: Optional[int] = None
    n_sec_max: Optional[float] = None
    n_random_frames: Optional[int] = None
    min_n_frames_per_scene: int = 3
    max_n_frames_per_scene: int = 20
    scene_length_threshold: float = 60