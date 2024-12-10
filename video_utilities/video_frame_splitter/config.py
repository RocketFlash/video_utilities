from dataclasses import dataclass, asdict, field
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)

@dataclass
class VideoFrameSplitterConfig():
    start_idx: int = 0
    start_sec: Optional[float] = 0
    frame_interval: int = 1
    frame_interval_sec: Optional[float] = 1
    frame_max_size: Optional[int] = 512
    n_frames_max: Optional[int] = None
    n_sec_max: Optional[float] = None