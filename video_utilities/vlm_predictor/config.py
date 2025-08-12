import torch
from dataclasses import dataclass, field
from typing import (
    Union,
    Optional,
    List,
    Dict
)

DEFAULT_GENERATION_PARAMS = dict(
    # min_length=20,
    # max_length=512,
    # length_penalty=1,
    # repetition_penalty=1.5,
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
class VLMPredictorConfig():
    model_name: str = 'Salesforce/blip2-opt-2.7b'
    gguf_file: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    input_content_type: str = 'images' # one of ['images', 'video'] 
    generation_params: dict = field(default_factory=lambda: DEFAULT_GENERATION_PARAMS)
    prompt_template: str = 'Question: {} Answer:'
    output_template: str = '{}: {}\n'
    additional_params: dict = field(default_factory=lambda: {})
    generate_instructions: bool = True
    use_outlines_model: bool = False