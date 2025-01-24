import torch
from dataclasses import dataclass, field
from typing import (
    Union,
    Optional,
    List,
    Dict
)


DEFAULT_GENERATION_PARAMS = dict(
    min_length=20,
    max_length=512,
    length_penalty=1,
    repetition_penalty=1.5,
    temperature=1
)
DEFAULT_QUESTIONS = [
    'What is this?'
]
DEFAULT_TAGS = dict(
    location=dict(
        description='scene of action, location',
        tags=[
            'indoor',
            'outdoor',
        ]
    ),
)


@dataclass
class FrameCaptionerConfig():
    model_name: str = 'Salesforce/blip2-opt-2.7b'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    questions: Union[List[str], Dict[str, Dict]] = field(default_factory=lambda: DEFAULT_QUESTIONS) 
    tags: Union[Dict[str, Dict[str, str]], List[str]] = field(default_factory=lambda: DEFAULT_TAGS)
    prompt: str = 'In this video frame'
    mode: str = 'simple' # ['simple', 'prompted', 'qa', 'chat']
    generation_params: dict = field(default_factory=lambda: DEFAULT_GENERATION_PARAMS)
    qa_input_template: str = 'Question: {} Answer:'
    tagging_input_template: str = 'Based on the visual content of the video frame, choose the tags that best describe {} what is shown. If no tags apply, state "None". \n\nList of tags: \n{}'
    output_template: str = '{}: {}\n'
    additional_params: dict = field(default_factory=lambda: {})