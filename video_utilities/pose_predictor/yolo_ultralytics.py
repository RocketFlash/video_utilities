import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO
from .pose_predictor import (
    PosePredictor,
    PosePredictorFrameOutputResult
)
from typing import (
    Union, 
    Optional,
    List, 
    Dict,
    Tuple, 
    Set
)
from .config import YOLOUltralyticsConfig

    
    
class YOLOUltralyticsPredictor(PosePredictor):
    landmarks_connections = set([
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6)
    ])
    landmarks_left_idxs = frozenset([
        3, 5, 7, 9, 11, 13, 15
    ])
    
    landmarks_right_idxs = frozenset([
        4, 6, 8, 10, 12, 14, 16
    ])

    def set_params_from_config(
        self, 
        config: YOLOUltralyticsConfig
    ):
        self.model_name = config.model_name
        self.conf = config.conf
        self.iou = config.iou
        self.imgsz = config.imgsz
        self.half = config.half
        self.device = config.device
        self.batch = config.batch
        self.model = YOLO(self.model_name)
        if isinstance(config.class_idx, int):
            config.class_idx = [config.class_idx]
        self.class_idx = config.class_idx

    
    def __call__(
        self,
        frames
    ):
        video_output_results = []
        
        for frame in frames:
            if isinstance(frame, np.ndarray):
                image = frame
                frame_idx = None
                timestamp = None
            else:
                image = frame.image
                frame_idx = frame.idx
                timestamp = frame.timestamp
                
            results = self.model.track(
                image,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                half=self.half,
                device=self.device,
                batch=self.batch,
                verbose=False
            )[0]

            if results:
                landmarks_2d = {}
                kpts_loc = results.keypoints.xyn.cpu().numpy()
                kpts_conf = results.keypoints.conf.cpu().numpy()
                bbox_cls = results.boxes.cls.cpu().numpy()
                bbox_confs = results.boxes.conf.cpu().numpy()
                bbox_ids = results.boxes.id.cpu().numpy()
                n_persons = kpts_loc.shape[0]

                for i in range(n_persons):
                    person_kpts_loc = kpts_loc[i]
                    person_kpts_conf = kpts_conf[i]
                    person_bbox_cls = bbox_cls[i]
                    person_bbox_conf = bbox_confs[i]
                    person_bbox_id = int(bbox_ids[i])
                    if person_bbox_cls not in self.class_idx:
                        continue

                    person_landmarks_2d_list = []

                    for person_kpt_idx, person_kpt_loc in enumerate(person_kpts_loc):
                        landmark_2d = dict(
                            x=person_kpt_loc[0],
                            y=person_kpt_loc[1],
                            visibility=person_kpts_conf[person_kpt_idx]
                        )
                        person_landmarks_2d_list.append(landmark_2d)
                        
                    landmarks_2d[person_bbox_id] = dict(
                        landmarks=person_landmarks_2d_list,
                        cls=person_bbox_cls,
                        conf=person_bbox_conf
                    )
            else:
                landmarks_2d = None

            landmarks_3d = None
            segmentation_mask = None

            video_frame_output_result = PosePredictorFrameOutputResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                landmarks_2d=landmarks_2d,
                landmarks_3d=None,
                segmentation_mask=None
            )
            video_output_results.append(video_frame_output_result)
            
        return video_output_results