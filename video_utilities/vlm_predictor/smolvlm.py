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
    AutoModelForImageTextToText
)
from .vlm_predictor import VLMPredictor


class SmolVLM_VLMPredictor(VLMPredictor):
    image_content_template = {
        "type": "image",
        "image": "",
    }
    video_content_template = {
        "type": "video",
        "video": [],
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
            attn_implementation = 'sdpa'

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
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

        if self.input_content_type=='video':
            video_content = copy.deepcopy(self.video_content_template)
            video_content['video'] = images
            message["content"].append(video_content)
        else:
            for image in images:
                image_content = copy.deepcopy(self.image_content_template)
                image_content['image'] = image
                message["content"].append(image_content)

        text_content = copy.deepcopy(self.text_content_template)
        text_content['text'] = text

        message["content"].append(text_content)
        messages = [message]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.dtype)
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