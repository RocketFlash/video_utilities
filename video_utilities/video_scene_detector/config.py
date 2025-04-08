from dataclasses import dataclass, asdict, field
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)

@dataclass
class VideoSceneDetectorConfig():
    threshold: float = 27.0
    min_scene_len: int = 15
    show_progress: bool = True
    backend: str = 'opencv'