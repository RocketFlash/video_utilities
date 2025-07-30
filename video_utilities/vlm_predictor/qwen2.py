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
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from .vlm_predictor import VLMPredictor


class Qwen2VL_VLMPredictor(VLMPredictor):
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
    system_message_template = {
        "role": "system",
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

        model_name_lower = model_name.lower()
        if 'qwen2.5' in model_name_lower:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                attn_implementation=attn_implementation,
                device_map=self.device,
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                attn_implementation=attn_implementation,
                device_map=self.device,
            )

        if 'chat_template' in self.additional_params:
            if self.additional_params['chat_template']:
                processor.chat_template = self.additional_params['chat_template']

        return model, processor


    def prepare_prompt_and_messages(
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
            video_content['fps'] = self.fps
            message["content"].append(video_content)
        else:
            for image in images:
                image_content = copy.deepcopy(self.image_content_template)
                image_content['image'] = image
                message["content"].append(image_content)

        text_content = copy.deepcopy(self.text_content_template)
        text_content['text'] = text
        message["content"].append(text_content)
        
        messages = []
        if self.system_prompt is not None:
            system_message = copy.deepcopy(self.system_message_template)
            system_message['content'] = self.system_prompt
            messages.append(system_message)

        messages.append(message)

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt, messages


    def process_visual_data_and_text(
        self, 
        visual_data: Union[List[np.ndarray], np.ndarray], 
        text: str
    ):
        prompt, messages = self.prepare_prompt(
            visual_data=visual_data, 
            text=text
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True
        )

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            fps=self.fps,
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


    def generate_output_outlines(
        self, 
        visual_data, 
        prompt
    ):
        if self.outlines_generator is not None:
            prompt, messages = self.prepare_prompt_and_messages(
                visual_data=visual_data, 
                text=prompt
            )

            max_new_tokens = self.generation_params['max_new_tokens'] if 'max_new_tokens' in self.generation_params else 1024

            result = self.outlines_generator(
                {
                    "text": prompt, 
                    "images": visual_data
                },
                max_new_tokens=max_new_tokens
            )
        else:
            print('There is no outlines generator')
            result = None
        return result
    

    def generate_output_outlines_batch(
        self, 
        visual_data, 
        prompt
    ):
        prompt_samples = []
        if self.outlines_generator is not None:
            for visual_data_sample in visual_data:
                prompt_sample, messages = self.prepare_prompt_and_messages(
                    visual_data=visual_data_sample, 
                    text=prompt
                )
                prompt_samples.append(prompt_sample)

            input_data = {
                "text": prompt_samples, 
                "images": visual_data
            }

            max_new_tokens = self.generation_params['max_new_tokens'] if 'max_new_tokens' in self.generation_params else 1024

            result = self.outlines_generator(
                input_data,
                max_new_tokens=max_new_tokens
            )
        else:
            print('There is no outlines generator')
            result = None
        return result