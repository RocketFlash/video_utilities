from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
)
from .image_captioner import ImageCaptioner


class BLIP2ImageCaptioner(ImageCaptioner):
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