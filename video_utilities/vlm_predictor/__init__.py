from .vlm_predictor import VLMPredictor
from .blip2 import BLIP2_VLMPredictor
from .pixtral import Pixtral_VLMPredictor
from .ovis import Ovis_VLMPredictor
from .llama3 import Llam3VL_VLMPredictor
from .qwen2 import Qwen2VL_VLMPredictor
from .paligemma import PaliGemma2_VLMPredictor
from .config import VLMPredictorConfig
from .internvl import InternVL_VLMPredictor
# from .deepseek_vl2 import DeepSeekVL2VLMPredictor


def get_vlm_predictor(
    config : VLMPredictorConfig = None
):
    model_name_lower = config.model_name.lower()
    if 'paligemma' in model_name_lower:
        return PaliGemma2_VLMPredictor(config)
    # elif 'deepseek-vl2' in model_name_lower:
    #     return DeepSeekVL2VLMPredictor(config)
    elif 'qwen2' in model_name_lower:
        return Qwen2VL_VLMPredictor(config)
    elif 'llama' in model_name_lower:
        return Llam3VL_VLMPredictor(config)
    elif 'ovis' in model_name_lower:
        return Ovis_VLMPredictor(config)
    elif 'pixtral' in model_name_lower:
        return Pixtral_VLMPredictor(config)
    elif 'internvl' in model_name_lower:
        return InternVL_VLMPredictor(config)
    else:
        return BLIP2_VLMPredictor(config)