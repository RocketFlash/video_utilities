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
class VideoFrameOutput:
    image: np.ndarray
    idx: int
    timestamp: float
    scene_id: Optional[int]


class VideoCaptioner:
    def __init__(
        self,
        frame_captioner,
        frame_output_processor = None,
    ):
        self.set_frame_captioner(frame_captioner)
        self.set_frame_output_processor(frame_output_processor)

    def set_frame_captioner(self, frame_captioner):
        self.frame_captioner = frame_captioner

    def set_frame_output_processor(self, frame_output_processor):
        self.frame_output_processor = frame_output_processor

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
            outputs = self.frame_captioner(frame)
            if self.frame_output_processor is not None:
                outputs_processed = {}
                for k, v in outputs.items():
                    if isinstance(expected_output_type, str):
                        exp_out_type = expected_output_type
                    else:
                        exp_out_type = expected_output_type[k]

                    outputs_processed[k] = self.frame_output_processor(
                        v,
                        expected_output_type=exp_out_type
                    )
                outputs = outputs_processed
            frames_results.append(outputs)

        return frames_results