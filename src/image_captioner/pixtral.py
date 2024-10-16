import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration
)
from transformers import BitsAndBytesConfig
from .image_captioner import ImageCaptioner


class PixtralImageCaptioner(ImageCaptioner):
    def get_model_and_processor(
        self,
        model_name: str,
    ):
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # load_in_4bit=True,
                # bnb_4bit_compute_dtype=torch.bfloat16,
                # bnb_4bit_quant_type="nf4"
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name, 
                device_map=self.device, 
                torch_dtype=self.dtype, 
                quantization_config=quantization_config)
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                use_safetensors=True,
                device_map=self.device
            ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor


    def process_image_and_text(self, image, text):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        if text is None:
            text = 'Caption this image:'

        prompt = f"<s>[INST]{text}\n[IMG][/INST]"
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        return inputs

    def decode_ids(
        self,
        generated_ids,
        prompt_len,
        inputs=None
    ):
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0][prompt_len:]
        return generated_text