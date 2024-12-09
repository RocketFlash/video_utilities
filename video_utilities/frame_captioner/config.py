import torch
from dataclasses import dataclass, field

DEFAULT_GENERATION_PARAMS = dict(
    min_length=20,
    max_length=512,
    length_penalty=1,
    repetition_penalty=1.5,
    temperature=1
)

@dataclass
class FrameCaptionerConfig():
    model_name: str = 'Salesforce/blip2-opt-2.7b'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    questions=[None]
    tags=[]
    prompt: str = 'In this video frame'
    mode: str = 'simple' # ['simple', 'prompted', 'qa', 'chat']
    use_quantization: bool = False
    generation_params: dict = field(default_factory=lambda: DEFAULT_GENERATION_PARAMS)
    qa_input_template: str = 'Question: {} Answer:'
    tagging_input_template = 'Based on the visual content of the video frame, choose the tags that best describe {} what is shown. Provide the result as a list of strings inside [] brackets in which the values ​​are separated by commas. If no tags apply, state "None". \n\nList of tags: \n{}'
    tags_desc = None
    output_template = '{}\n {}\n'
    attn_implementation: str = 'sdpa'