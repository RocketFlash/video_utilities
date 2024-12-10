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
        frame_captioner,
        vlm_output_processor = None,
        vlm_output_validator = None
    ):
        self.set_frame_captioner(frame_captioner)
        self.set_vlm_output_processor(vlm_output_processor)
        self.set_vlm_output_validator(vlm_output_validator)

    def set_frame_captioner(self, frame_captioner):
        self.frame_captioner = frame_captioner

    def set_vlm_output_processor(self, vlm_output_processor):
        self.vlm_output_processor = vlm_output_processor

    def set_vlm_output_validator(self, vlm_output_validator):
        self.vlm_output_validator = vlm_output_validator

    def set_mode(self, mode):
        self.frame_captioner.set_mode(mode)

    def set_prompt(self, prompt):
        self.frame_captioner.set_prompt(prompt)

    def set_tags(self, tags):
        self.frame_captioner.set_tags(tags)

    def set_tags_desc(self, tags_desc):
        self.frame_captioner.set_tags_desc(tags_desc)

    def set_questions(self, questions):
        self.frame_captioner.set_questions(questions)

    def set_qa_input_template(self, qa_input_template):
        self.frame_captioner.set_qa_input_template(qa_input_template)

    def set_tagging_input_template(self, tagging_input_template):
        self.frame_captioner.set_tagging_input_template(tagging_input_template)

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

            outputs = self.frame_captioner(image)

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
                if self.frame_captioner.mode == 'tagging':
                    outputs = {}
                    for k, v in outputs_processed.items():
                        outputs[k] = v[k]
                elif self.frame_captioner.mode == 'tagging_merged':
                    outputs = outputs_processed['predictions']
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