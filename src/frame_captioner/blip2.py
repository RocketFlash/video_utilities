from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
)
from .frame_captioner import FrameCaptioner


class BLIP2FrameCaptioner(FrameCaptioner):
    def get_model_and_processor(
        self,
        model_name: str,
    ):
        processor = AutoProcessor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        ).eval()
        model = model.to(self.device)
        return model, processor