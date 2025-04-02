import numpy as np
import copy
from PIL import Image
from typing import (
    Union,
    Optional,
    List,
    Dict
)
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq
)
from .vlm_predictor import VLMPredictor


class SmolVLM_VLMPredictor(VLMPredictor):
    image_content_template = {
        "type": "image",
    }
    text_content_template = {
        "type": "text",
        "text": "Describe this image."
    }
    message_template = {
        "role": "user",
        "content": [],
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
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=self.dtype,
            _attn_implementation=attn_implementation,
        ).to(self.device)

        return model, processor


    def process_visual_data_and_text(
        self, 
        visual_data: Union[List[np.ndarray], np.ndarray], 
        text: str
    ):
        message = copy.deepcopy(self.message_template)

        images = []
        if isinstance(visual_data, list):
            for image in visual_data:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                images.append(image)
        else:
            if isinstance(visual_data, np.ndarray):
                image = Image.fromarray(visual_data)
            else:
                image = visual_data
            images = [image]

        for image in images:
            image_content = copy.deepcopy(self.image_content_template)
            message["content"].append(image_content)

        text_content = copy.deepcopy(self.text_content_template)
        text_content['text'] = text
        message["content"].append(text_content)
        messages = [message]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs
    

    def decode_ids(
        self,
        generated_ids,
        prompt_len=None,
        inputs=None
    ):
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        generated_text = generated_text.strip()[prompt_len:]
        return generated_text