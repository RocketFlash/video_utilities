from .blip2 import BLIP2FrameCaptioner
from .pixtral import PixtralFrameCaptioner
from .ovis import OvisFrameCaptioner
from .llama3 import Llam3VLFrameCaptioner
from .qwen2 import Qwen2VLFrameCaptioner
from .paligemma import PaliGemma2FrameCaptioner


def get_captioner(
    model_name='Salesforce/blip2-opt-2.7b',
    **frame_captioner_params
):
    model_name_lower = model_name.lower()
    if 'paligemma' in model_name_lower:
        return PaliGemma2FrameCaptioner(model_name, **frame_captioner_params)
    elif 'qwen2' in model_name_lower:
        return Qwen2VLFrameCaptioner(model_name, **frame_captioner_params)
    elif 'llama' in model_name_lower:
        return Llam3VLFrameCaptioner(model_name, **frame_captioner_params)
    elif 'ovis' in model_name_lower:
        return OvisFrameCaptioner(model_name, **frame_captioner_params)
    elif 'pixtral' in model_name_lower:
        return PixtralFrameCaptioner(model_name, **frame_captioner_params)
    else:
        return BLIP2FrameCaptioner(model_name, **frame_captioner_params)