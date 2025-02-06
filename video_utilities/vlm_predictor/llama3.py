import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)
from .vlm_predictor import VLMPredictor


class Llam3VL_VLMPredictor(VLMPredictor):
    message_template = {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": "Describe this image."
            }
    ]}

    def get_model_and_processor(
        self,
        model_name: str,
    ):
        processor = AutoProcessor.from_pretrained(model_name)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )

        return model, processor

    def process_image_and_text(self, image, text):
        message = self.message_template.copy()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        message['content'][1]['text']  = text
        messages = [message]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=image,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device, self.dtype)
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
        )[0].strip()
        return generated_text