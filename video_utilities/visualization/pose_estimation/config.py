from dataclasses import dataclass, asdict, field
from typing import (
    Union, 
    Optional,
    List, 
    Dict,
    Tuple, 
    Set
)


@dataclass
class Pose2DVisualizerConfig:
    left_color: Tuple[int, int, int] = (0, 0, 255) 
    right_color: Tuple[int, int, int] = (255, 0, 0) 
    joint_color: Tuple[int, int, int] = (0, 255, 255) 
    link_color: Tuple[int, int, int] = (0, 255, 0)
    joint_radius: int = 5
    link_thickness: int = 2
    visibility_threshold: float = 0.5
    id_color: Tuple[int, int, int] = (255, 0, 0)
    id_font_scale: float = 0.5
    id_thickness: int = 2