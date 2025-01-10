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
class YOLOUltralyticsConfig:
    r"""
    Parameters description could be found here:
    https://docs.ultralytics.com/modes/predict/#inference-sources
    """
    model_name: str = "yolo11n-pose.pt"
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    half: bool = False
    device: str = "cuda:0"
    batch: int = 1
    agnostic_nms: bool = False
    class_idx: Union[int, List[int]] = 0