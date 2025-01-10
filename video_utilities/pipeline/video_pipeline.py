import cv2
import json
import numpy as np
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)
from pathlib import Path
from dataclasses import dataclass
from ..video_frame_splitter import (
    VideoFramesData,
    VideoFrameSplitter
)
from ..video_scene_detector import (
    SceneData,
    VideoSceneDetector
)
from ..video_captioner import (
    VideoFrameOutputResult,
    VideoCaptioner
)
from ..video_results_aggregator import VideoResultsAggregator


@dataclass
class VideoPipelineResults:
    video_frames_data: VideoFramesData
    scene_list: List[SceneData]
    frames_results: List[VideoFrameOutputResult]


class VideoPipeline:
    def __init__(
        self,
        video_frame_splitter: VideoFrameSplitter,
        video_scene_detector: Optional[VideoSceneDetector],
        video_captioner: VideoCaptioner,
        video_results_aggregator: Optional[VideoResultsAggregator]
    ):
        self.video_frame_splitter = video_frame_splitter
        self.video_captioner = video_captioner
        self.video_scene_detector = video_scene_detector
        self.video_results_aggregator = video_results_aggregator

    def __call__(
        self,
        video_path,
        n_frames_per_scene = 1,
        expected_output_type='dict[list[str]]',
        save_images = False,
        save_dir = None,
        save_json_name = "predictions.json"
    ):
        if self.video_scene_detector is not None:
            scene_list = self.video_scene_detector(video_path)
        else:
            scene_list = None

        video_frames_data = self.video_frame_splitter(
            video_path,
            scene_list=scene_list,
            n_frames_per_scene=n_frames_per_scene
        )

        frames_results = self.video_captioner(
            video_frames_data.frames,
            expected_output_type=expected_output_type
        )
        
        if save_dir is not None:
            video_name = Path(video_path).stem
            save_dir_video = Path(save_dir) / video_name
            save_dir_video.mkdir(exist_ok=True)

            save_path_json = Path(save_dir_video) / save_json_name
            
            if self.video_results_aggregator is not None:
                video_results_json = self.video_results_aggregator.to_json(frames_results)
                with open(save_path_json, "w") as outfile:
                    outfile.write(video_results_json)

            if save_images:
                for frame in video_frames_data.frames:
                    save_name_frame = f'frame_{frame.idx}.jpg'
                    save_path_frame = save_dir_video / save_name_frame
                    image = frame.image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                    cv2.imwrite(save_path_frame, image)

        video_results = VideoPipelineResults(
            video_frames_data=video_frames_data,
            scene_list=scene_list,
            frames_results=frames_results
        )
        return video_results