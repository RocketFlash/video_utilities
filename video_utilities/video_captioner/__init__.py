from tqdm.auto import tqdm
import numpy as np
from dataclasses import dataclass
from typing import (
    Union,
    Optional,
    List,
    Dict
)


@dataclass
class VideoFrameOutputResult:
    frame_idx: int
    timestamp: float
    scene_id: Optional[int]
    outputs: dict


class VideoCaptioner:
    def __init__(
        self,
        vlm_predictor,
        vlm_output_processor = None,
        vlm_output_validator = None
    ):
        self.set_vlm_predictor(vlm_predictor)
        self.set_vlm_output_processor(vlm_output_processor)
        self.set_vlm_output_validator(vlm_output_validator)

    def set_vlm_predictor(self, vlm_predictor):
        self.vlm_predictor = vlm_predictor

    def set_vlm_output_processor(self, vlm_output_processor):
        self.vlm_output_processor = vlm_output_processor

    def set_vlm_output_validator(self, vlm_output_validator):
        self.vlm_output_validator = vlm_output_validator

    def set_mode(self, mode):
        self.vlm_predictor.set_mode(mode)

    def set_prompt(self, prompt):
        self.vlm_predictor.set_prompt(prompt)

    def set_tags(self, tags):
        self.vlm_predictor.set_tags(tags)

    def set_tags_desc(self, tags_desc):
        self.vlm_predictor.set_tags_desc(tags_desc)

    def set_questions(self, questions):
        self.vlm_predictor.set_questions(questions)

    def set_qa_input_template(self, qa_input_template):
        self.vlm_predictor.set_qa_input_template(qa_input_template)

    def set_tagging_input_template(self, tagging_input_template):
        self.vlm_predictor.set_tagging_input_template(tagging_input_template)

    def __call__(
        self,
        frames,
        expected_output_type: Union[str, Dict[str, str]] = 'str'
    ):
        frames_results = []

        for frame in tqdm(frames):
            if isinstance(frame, np.ndarray):
                image = frame
                frame_idx = None
                timestamp = None
                scene_id  = None
            else:
                image = frame.image
                frame_idx = frame.idx
                timestamp = frame.timestamp
                scene_id  = frame.scene_id

            outputs = self.vlm_predictor(image)
              
            if self.vlm_output_processor is not None:
                outputs_processed = {}
                for k, v in outputs.items():
                    if isinstance(expected_output_type, str):
                        exp_out_type = expected_output_type
                    else:
                        exp_out_type = expected_output_type[k]

                    output_processed = self.vlm_output_processor(
                        v,
                        expected_output_type=exp_out_type
                    )
                    if self.vlm_output_validator is not None:
                        output_processed = self.vlm_output_validator(output_processed)
                    outputs_processed[k] = output_processed 
                if self.vlm_predictor.mode.split('_')[0] in ['tagging', 'qa']:
                    if 'merged' in self.vlm_predictor.mode:
                        outputs = outputs_processed['predictions']
                    else:
                        outputs = {}
                        for k, v in outputs_processed.items():
                            outputs[k] = v[k]
                else:
                    outputs = outputs_processed

            video_frame_output_result = VideoFrameOutputResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                scene_id=scene_id,
                outputs=outputs
            )
            frames_results.append(video_frame_output_result)

        return frames_results