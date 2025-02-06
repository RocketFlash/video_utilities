import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import (
    DeepseekVLV2Processor, 
    DeepseekVLV2ForCausalLM
)

from .vlm_predictor import VLMPredictor


class DeepSeekVL2_VLMPredictor(VLMPredictor):
    message_template = [
        {
            "role": "<|User|>",
            "content": "<image>\n{}",
            "images": [],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]


    def get_model_and_processor(
        self,
        model_name: str,
    ):
        processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = processor.tokenizer

        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        model = model.to(torch.bfloat16).to(self.device).eval()

        return model, processor


    def generate_output(self, image, prompt):
        prompt_len = len(prompt) if prompt is not None else None
        prepare_inputs, inputs = self.process_image_and_text(
            image=image,
            text=prompt
        )
        with torch.inference_mode():
            generated_ids = self.model.language_model.generate(
                inputs_embeds=inputs,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.generation_params
            )

        generated_text = self.decode_ids(
            generated_ids,
            prompt_len=prompt_len,
            inputs=inputs
        )
        return generated_text
    

    def process_image_and_text(self, image, text):
        message = self.message_template.copy()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        message[0]['content'] = message[0]['content'].format(text)
        
        prepare_inputs = self.processor(
            conversations=message,
            images=[image],
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        inputs = self.model.prepare_inputs_embeds(**prepare_inputs)
        return prepare_inputs, inputs


    def decode_ids(
        self,
        generated_ids,
        prompt_len,
        inputs=None
    ):
        generated_text = self.tokenizer.decode(
            generated_ids[0].cpu().tolist(), 
            skip_special_tokens=True
        )
        return generated_text