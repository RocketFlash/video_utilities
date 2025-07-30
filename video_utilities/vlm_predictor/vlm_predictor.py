import torch
import numpy as np
from typing import (
    Union,
    Optional,
    List,
    Dict
)
from pydantic import BaseModel
from typing import (
    Type, 
    TypeVar
)
from .config import VLMPredictorConfig
from ..utils import generate_instruction_str

try:
    import outlines
    OUTLINES_INSTALLED = True
except ImportError:
    OUTLINES_INSTALLED = False

T = TypeVar('T', bound=BaseModel)

class VLMPredictor:
    def __init__(
        self,
        config = None,
    ):
        if config is None:
            config = self.get_default_config()
        self.config = config
        self.set_params_from_config(config)

        model, processor = self.get_model_and_processor(
            model_name=self.model_name
        )
        self.model = model
        self.processor = processor

        self.prompt = None
        self.system_prompt = None
        self.queries_dict = None
        self.output_schema = None
        self.outlines_generator = None
        self.outlines_model = None
        self.fps = 2

        if self.use_outlines_model:
            if OUTLINES_INSTALLED:
                self.outlines_model = outlines.models.from_transformers(
                    self.model, 
                    self.processor
                )
            else:
                print('Outlines library is not installed.')
        

    def get_default_config(self):
        return VLMPredictorConfig()


    def set_params_from_config(
        self, 
        config: VLMPredictorConfig
    ):
        for key, value in vars(config).items():
            setattr(self, key, value)
    

    def get_model_and_processor(
        self,
        model_name: str,
    ):
        return None, None


    def construct_queries_str(self):
        if self.queries_dict is None or len(self.queries_dict)==0:
            return ''

        query_strings_list = []
        for query_category, query_category_dict in self.queries_dict.items():
            query = query_category_dict['query']

            instruction_str = ''
            description_str = ''

            if 'instruction' in query_category_dict:
                instruction_str = query_category_dict['instruction']
            else:
                if self.generate_instructions: 
                    instruction_str = generate_instruction_str(query_category_dict)

            if 'description' in query_category_dict:
                description_str = query_category_dict['description']
            
            category_input_str = f'category name: {query_category}\n'
            if description_str:
                category_input_str += f'category description: {description_str}\n'
            
            category_input_str += f'query: {query}\n'
            if instruction_str:
                category_input_str += f'instruction: {instruction_str}\n'
            
            query_strings_list.append(category_input_str)

        queries_str = '\n'.join(query_strings_list)
        return queries_str


    def set_input_content_type(
        self, 
        input_content_type: str
    ):
        self.input_content_type = input_content_type


    def set_system_prompt(
        self, 
        system_prompt: str
    ):
        self.system_prompt = system_prompt


    def set_prompt(
        self, 
        prompt: str
    ):
        self.prompt = prompt

    
    def set_output_schema(
        self, 
        output_schema: T
    ):
        self.output_schema = output_schema
        if self.outlines_model is not None:
            self.outlines_generator = outlines.Generator(
                self.outlines_model, 
                self.output_schema
            )


    def set_fps(
        self, 
        fps: int
    ):
        self.fps = fps


    def set_generation_params(
        self, 
        generation_params: Dict
    ):
        self.generation_params = generation_params


    def set_structured_prompt(
        self, 
        queries_dict: Dict
    ):
        self.queries_dict = queries_dict
        queries_str = self.construct_queries_str()
        self.prompt = self.prompt_template.format(queries=queries_str)


    def set_prompt_template(
        self, 
        prompt_template: str
    ):
        self.prompt_template = prompt_template


    def set_output_template(
        self, 
        output_template: str
    ):
        self.output_template = output_template


    def process_visual_data(
        self, 
        visual_data
    ):
        inputs_image = self.processor(
            images=visual_data,
            return_tensors="pt"
        ).to(self.model.device)
        return inputs_image


    def process_text(self, text):
        inputs_text = self.processor(
            text=text,
            return_tensors="pt"
        ).to(self.model.device)
        return inputs_text


    def process_visual_data_and_text(
        self, 
        visual_data, 
        text
    ):
        inputs = self.processor(
            images=visual_data,
            text=text,
            return_tensors="pt"
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
        generated_text = generated_text.strip()
        return generated_text


    def generate_output(
        self, 
        visual_data, 
        prompt
    ):
        prompt_len = len(prompt) if prompt is not None else None
        inputs = self.process_visual_data_and_text(
            visual_data=visual_data,
            text=prompt
        )
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_params
            )

        generated_text = self.decode_ids(
            generated_ids,
            prompt_len=prompt_len,
            inputs=inputs
        )
        return generated_text


    def generate_output_outlines(
        self, 
        visual_data, 
        prompt
    ):
        return None
    

    def generate_output_outlines_batch(
        self, 
        visual_data, 
        prompt
    ):
        return None


    def __call__(
        self, 
        visual_data: Union[List[List[np.ndarray]], List[np.ndarray], np.ndarray],
        batch_processing: bool = False
    ):
        if batch_processing and isinstance(visual_data, list):
            if self.use_outlines_model and OUTLINES_INSTALLED:
                return self.generate_output_outlines_batch(visual_data, self.prompt)
            else:
                return None
        else:
            if self.use_outlines_model and OUTLINES_INSTALLED:
                return self.generate_output_outlines(visual_data, self.prompt)
            else:
                return self.generate_output(visual_data, self.prompt)