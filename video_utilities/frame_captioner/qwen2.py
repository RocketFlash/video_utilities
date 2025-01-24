import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from .frame_captioner import FrameCaptioner


class Qwen2VLFrameCaptioner(FrameCaptioner):
    message_template = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "",
            },
            {
                "type": "text",
                "text": "Describe this image."
            },
        ],
    }

    def get_model_and_processor(
        self,
        model_name: str,
    ):
    
        if 'attn_implementation' in self.additional_params:
            attn_implementation = self.additional_params['attn_implementation']
        else:
            attn_implementation = 'flash_attention_2'

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
            device_map=self.device,
        )

        return model, processor

    def process_image_and_text(self, image, text):
        message = self.message_template.copy()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        message['content'][0]['image'] = image
        message['content'][1]['text']  = text
        messages = [message]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs


    def decode_ids(
        self,
        generated_ids,
        prompt_len,
        inputs=None
    ):
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        return generated_text