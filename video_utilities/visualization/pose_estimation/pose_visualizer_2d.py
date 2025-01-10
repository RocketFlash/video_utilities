import cv2
import numpy as np
from tqdm.auto import tqdm
from .config import Pose2DVisualizerConfig
from ...pose_predictor import PosePredictorFrameOutputResult
from ...video_frame_splitter import VideoFrame


from typing import (
    Union, 
    Optional,
    List, 
    Dict,
    Tuple, 
    Set
)


class Pose2DVisualizer:
    def __init__(
        self,
        connections: Set[Tuple[int, int]],
        left_body_idxs: Optional[Set[int]],
        right_body_idxs: Optional[Set[int]],
        config: Optional[Pose2DVisualizerConfig]
    ):
        self.connections = connections
        self.left_body_idxs = left_body_idxs
        self.right_body_idxs = right_body_idxs
        if config is None:
            config = Pose2DVisualizerConfig()
        self.config = config

        self.set_params_from_config(config=config)
        

    def set_params_from_config(
        self, 
        config: Pose2DVisualizerConfig
    ):
        self.left_color = config.left_color 
        self.right_color = config.right_color  
        self.joint_color = config.joint_color 
        self.link_color = config.link_color
        self.joint_radius = config.joint_radius
        self.link_thickness = config.link_thickness
        self.visibility_threshold = config.visibility_threshold
        self.id_color = config.id_color
        self.id_font_scale = config.id_font_scale
        self.id_thickness = config.id_thickness
        

    def draw_landmarks_2d(
        self, 
        image: np.ndarray, 
        landmarks_2d: Optional[Dict]
    ) -> np.ndarray:
        result_image = image.copy()
        if landmarks_2d is None:
            return result_image
        
        img_h, img_w = result_image.shape[:2]

        for person_id, person_data in landmarks_2d.items():
            person_landmarks_2d = person_data['landmarks']
            visible_landmarks = []
            
            for connection in self.connections:
                idx1, idx2 = connection
                if (idx1 < len(person_landmarks_2d) and idx2 < len(person_landmarks_2d) and
                    person_landmarks_2d[idx1].get('visibility', 1.0) >= self.visibility_threshold and
                    person_landmarks_2d[idx2].get('visibility', 1.0) >= self.visibility_threshold):
                    
                    x1, y1 = int(person_landmarks_2d[idx1]['x'] * img_w), int(person_landmarks_2d[idx1]['y'] * img_h)
                    x2, y2 = int(person_landmarks_2d[idx2]['x'] * img_w), int(person_landmarks_2d[idx2]['y'] * img_h)
                    cv2.line(result_image, (x1, y1), (x2, y2), self.link_color, self.link_thickness)
    
            for idx, landmark in enumerate(person_landmarks_2d):
                x, y = int(landmark['x'] * img_w), int(landmark['y'] * img_h)
                visibility = landmark.get('visibility', 1.0)
    
                if visibility >= self.visibility_threshold:
                    visible_landmarks.append((x, y))
                    if self.left_body_idxs is None and self.left_body_idxs is None:
                        color = self.joint_color
                    else:
                        color = self.left_color if idx in self.left_body_idxs else self.right_color
                    cv2.circle(result_image, (x, y), self.joint_radius, color, -1)

            if visible_landmarks:
                top_left_x = min(x for x, _ in visible_landmarks)
                top_left_y = min(y for _, y in visible_landmarks)
                id_position = (top_left_x, max(0, top_left_y - 30))
                cv2.putText(
                    result_image, 
                    f"ID: {person_id}", 
                    id_position, 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.id_font_scale, 
                    self.id_color, 
                    self.id_thickness, 
                    cv2.LINE_AA
                )

        return result_image


    def draw_landmarks_2d_frames(
        self,
        frames: Union[List[np.ndarray], List[VideoFrame]],
        landmarks_2d: Union[List[PosePredictorFrameOutputResult], List[Dict]]
    ):
        result_images = []
        for frame, landmarks_2d_frame in zip(frames, landmarks_2d):
            if isinstance(frame, VideoFrame):
                frame = frame.image

            if isinstance(landmarks_2d_frame, PosePredictorFrameOutputResult):
                landmarks_2d_frame = landmarks_2d_frame.landmarks_2d

            result_image = self.draw_landmarks_2d(
                image=frame,
                landmarks_2d=landmarks_2d_frame
            )

            result_images.append(result_image)
        return result_images


    def __call__(
        self,
        input_data: Union[List, np.ndarray],
        landmarks_2d: List
    ):
        if isinstance(input_data, list):
            result = self.draw_landmarks_2d_frames(
                frames=input_data,
                landmarks_2d=landmarks_2d
            )
        else:
            result = self.draw_landmarks_2d(
                image=input_data,
                landmarks_2d=landmarks_2d
            )
        return result