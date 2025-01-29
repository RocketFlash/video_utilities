from .frame_captioner import FrameCaptioner
from .blip2 import BLIP2FrameCaptioner
from .pixtral import PixtralFrameCaptioner
from .ovis import OvisFrameCaptioner
from .llama3 import Llam3VLFrameCaptioner
from .qwen2 import Qwen2VLFrameCaptioner
from .qwen2_5 import Qwen2_5VLFrameCaptioner
from .paligemma import PaliGemma2FrameCaptioner
from .config import FrameCaptionerConfig
from .internvl import InternVLFrameCaptioner
# from .deepseek_vl2 import DeepSeekVL2FrameCaptioner


def get_frame_captioner(
    config : FrameCaptionerConfig = None
):
    model_name_lower = config.model_name.lower()
    if 'paligemma' in model_name_lower:
        return PaliGemma2FrameCaptioner(config)
    # elif 'deepseek-vl2' in model_name_lower:
    #     return DeepSeekVL2FrameCaptioner(config)
    elif 'qwen2.5' in model_name_lower:
        return Qwen2_5VLFrameCaptioner(config)
    elif 'qwen2' in model_name_lower:
        return Qwen2VLFrameCaptioner(config)
    elif 'llama' in model_name_lower:
        return Llam3VLFrameCaptioner(config)
    elif 'ovis' in model_name_lower:
        return OvisFrameCaptioner(config)
    elif 'pixtral' in model_name_lower:
        return PixtralFrameCaptioner(config)
    elif 'internvl' in model_name_lower:
        return InternVLFrameCaptioner(config)
    else:
        return BLIP2FrameCaptioner(config)