import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
)
from .vlm_predictor import VLMPredictor


class Ovis_VLMPredictor(VLMPredictor):
    def get_model_and_processor(
        self,
        model_name: str,
    ):
        if 'multimodal_max_length' in self.additional_params:
            multimodal_max_length = self.additional_params['multimodal_max_length']
        else:
            multimodal_max_length = 8192

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            multimodal_max_length=multimodal_max_length,
            trust_remote_code=True
        ).eval()
        model = model.to(self.device)

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        processor = dict(
            text_tokenizer=text_tokenizer,
            visual_tokenizer=visual_tokenizer
        )
        self.generation_params['eos_token_id'] = model.generation_config.eos_token_id
        self.generation_params['pad_token_id'] = text_tokenizer.pad_token_id
        return model, processor


    def process_image_and_text(
        self, 
        image, 
        text
    ):
        query = f'<image>\n{text}'
        image = Image.fromarray(image)

        visual_tokenizer = self.processor['visual_tokenizer']
        text_tokenizer = self.processor['text_tokenizer']

        if 'max_partition' in self.additional_params:
            max_partition = self.additional_params['max_partition']
        else:
            max_partition = 9

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, 
            [image],
            max_partition=max_partition
        )

        attention_mask = torch.ne(
            input_ids,
            text_tokenizer.pad_token_id
        )
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)

        pixel_values = [
            pixel_values.to(
                dtype=visual_tokenizer.dtype,
                device=visual_tokenizer.device
            )
        ]

        inputs=dict(
            inputs=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )
        return inputs


    def decode_ids(
        self,
        generated_ids,
        prompt_len,
        inputs=None
    ):
        generated_text = self.processor['text_tokenizer'].decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        return generated_text