from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
from .frame_captioner import FrameCaptioner


class PaliGemma2FrameCaptioner(FrameCaptioner):
    def get_model_and_processor(
        self,
        model_name: str,
    ):
        processor = AutoProcessor.from_pretrained(model_name)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=self.dtype,
            revision="bfloat16"
        ).eval()

        return model, processor


    def decode_ids(
        self,
        generated_ids,
        prompt_len,
        inputs=None
    ):
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        generated_text = generated_text[prompt_len:].strip()
        return generated_text