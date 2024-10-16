from .blip2 import BLIP2ImageCaptioner
from .pixtral import PixtralImageCaptioner
from .ovis import OvisImageCaptioner
from .llama3 import Llam3VLImageCaptioner
from .qwen2 import Qwen2VLImageCaptioner
from .paligemma import PaliGemma2ImageCaptioner


def get_captioner(
    model_name='Salesforce/blip2-opt-2.7b',
    **image_captioner_params
):
    model_name_lower = model_name.lower()
    if 'paligemma' in model_name_lower:
        return PaliGemma2ImageCaptioner(model_name, **image_captioner_params)
    elif 'qwen2' in model_name_lower:
        return Qwen2VLImageCaptioner(model_name, **image_captioner_params)
    elif 'llama' in model_name_lower:
        return Llam3VLImageCaptioner(model_name, **image_captioner_params)
    elif 'ovis' in model_name_lower:
        return OvisImageCaptioner(model_name, **image_captioner_params)
    elif 'pixtral' in model_name_lower:
        return PixtralImageCaptioner(model_name, **image_captioner_params)
    else:
        return BLIP2ImageCaptioner(model_name, **image_captioner_params)