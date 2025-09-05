import cv2
from dataclasses import dataclass, asdict, field
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)
from enum import Enum

class VideoReaderType(Enum):
    """Enumeration of supported video reader backends."""
    DECORD = "decord"
    OPENCV = "opencv"
    AUTO = "auto"  # Try decord first, fallback to opencv

@dataclass
class VideoFrameSplitterConfig:
    """Configuration for the VideoFrameSplitter."""
    # Video reader settings
    video_reader_type: VideoReaderType = VideoReaderType.AUTO
    
    # Frame selection parameters
    start_idx: int = 0
    start_sec: Optional[float] = None
    frame_interval: int = 1
    frame_interval_sec: Optional[float] = None
    
    # Frame processing
    frame_max_size: Optional[int] = None
    maintain_aspect_ratio: bool = True
    interpolation_method: int = cv2.INTER_AREA
    
    # Selection limits
    n_frames_max: Optional[int] = None
    n_sec_max: Optional[float] = None
    n_random_frames: Optional[int] = None
    
    # Scene-based selection
    min_n_frames_per_scene: int = 3
    max_n_frames_per_scene: int = 20
    scene_length_threshold: float = 60.0
    
    # Performance and logging
    batch_size: int = 32  # For decord batch processing
    show_progress: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if self.frame_interval < 1:
            raise ValueError("frame_interval must be at least 1")
        if self.min_n_frames_per_scene < 1:
            raise ValueError("min_n_frames_per_scene must be at least 1")
        if self.max_n_frames_per_scene < self.min_n_frames_per_scene:
            raise ValueError("max_n_frames_per_scene must be >= min_n_frames_per_scene")
