from dataclasses import dataclass, asdict, field
from typing import (
    Union, 
    Optional,
    List, 
    Dict,
    Tuple, 
    Set
)
import numpy as np


class PosePredictor:
    def __init__(
        self,
        config
    ):
        self.config = config
        self.set_params_from_config(config=config)

    def set_params_from_config(
        self, 
        config
    ):
        pass


@dataclass
class PosePredictorFrameOutputResult:
    frame_idx: Optional[int]
    timestamp: Optional[float]
    landmarks_2d: Optional[Dict]
    landmarks_3d: Optional[Dict]
    segmentation_mask: Optional[np.ndarray]