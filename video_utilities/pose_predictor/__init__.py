from .config import (
    YOLOUltralyticsConfig
)
from .pose_predictor import (
    PosePredictorFrameOutputResult,
    PosePredictor
)
from .yolo_ultralytics import YOLOUltralyticsPredictor
from typing import (
    Union, 
    Optional,
    List, 
    Dict,
    Tuple, 
    Set
)


def get_pose_predictor(
    config : YOLOUltralyticsConfig = None
):
    model_name_lower = config.model_name.lower()
    if 'yolo' in model_name_lower:
        return YOLOUltralyticsPredictor(config)
    else:
        return None