import torch
import numpy as np
from typing import (
    Union,
    Optional,
    List,
    Dict
)
from .config import VLMPredictorConfig
from ..utils import generate_instruction_str


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

        self.update_qa_categories_input_str_dict()
        self.update_tagging_categories_input_str_dict()


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


    def update_qa_categories_input_str_dict(self):
        question_categories_dict = self.questions

        if isinstance(question_categories_dict, list):
            questions_list = question_categories_dict
            question_categories_dict = {}

            for i, question in enumerate(questions_list):
                question_category = f'question_{i}'
                question_categories_dict[question_category] = dict(
                    question=question,
                    expected_output_type='str',
                )
        
        qa_categories_input_str_dict = {}
        for question_category, question_category_dict in question_categories_dict.items():
            question = question_category_dict['question']

            instruction_str = ''
            description_str = ''

            if 'instruction' in question_category_dict:
                instruction_str = question_category_dict['instruction']
            else:
                if self.generate_instructions: 
                    instruction_str = generate_instruction_str(question_category_dict)

            if 'description' in question_category_dict:
                description_str = question_category_dict['description']
            

            category_input_str = f'category name: {question_category}\n'
            if description_str:
                category_input_str += f'category description: {description_str}\n'
            
            category_input_str += f'question: {question}\n'
            if instruction_str:
                category_input_str += f'instruction: {instruction_str}\n'
                
            qa_categories_input_str_dict[question_category] = category_input_str

        self.qa_categories_input_str_dict = qa_categories_input_str_dict


    def update_tagging_categories_input_str_dict(self):
        tagging_categories_dict = self.tags

        if isinstance(tagging_categories_dict, list):
            tagging_categories_dict = dict(
                general=dict(
                    tags=tagging_categories_dict,
                    description=''
                )
            )

        tagging_categories_input_str_dict = {} 
        for tag_category, tag_category_dict in tagging_categories_dict.items():
            tag_list = tag_category_dict['tags']
            tags_cat_desc = tag_category_dict['description']
            tag_names_str = ', '.join(tag_list)

            category_input_str = (
                f'category name: {tag_category}\n' 
                f'category description: {tags_cat_desc}\n' 
                f'available tags: [{tag_names_str}]\n'
            )
            tagging_categories_input_str_dict[tag_category] = category_input_str
        self.tagging_categories_input_str_dict = tagging_categories_input_str_dict


    def set_mode(self, mode):
        self.mode = mode
   

    def set_input_content_type(self, input_content_type):
        self.input_content_type = input_content_type


    def set_prompt(self, prompt):
        self.prompt = prompt


    def set_tags(self, tags):
        self.tags = tags
        self.update_tagging_categories_input_str_dict()


    def set_questions(self, questions):
        self.questions = questions
        self.update_qa_categories_input_str_dict()


    def set_qa_input_template(self, qa_input_template):
        self.qa_input_template = qa_input_template


    def set_tagging_input_template(self, tagging_input_template):
        self.tagging_input_template = tagging_input_template


    def set_output_template(self, output_template):
        self.output_template = output_template


    def set_pose_predictor(self, pose_predictor):
        self.pose_predictor = pose_predictor


    def process_visual_data(self, visual_data):
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


    def simple_captioning(self, visual_data):
        output = self.generate_output(visual_data, None)
        outputs = {'caption' : output}
        return outputs


    def prompted_captioning(self, visual_data):
        prompts = self.prompt
        if not isinstance(prompts, list):
            prompts = [prompts]

        outputs = {}
        for prompt in prompts:
            output = self.generate_output(visual_data, prompt)
            outputs[prompt] = output

        return outputs


    def tagging(self, visual_data):
        outputs = {}
        if self.mode == 'tagging_merged':
            categories_input_str_list = list(self.tagging_categories_input_str_dict.values())
            categories_input_str = '\n'.join(categories_input_str_list)
            prompt = self.tagging_input_template.format(input=categories_input_str)
            output = self.generate_output(visual_data, prompt)
            outputs['predictions'] = output
        else:
            for tags_category, category_input_str in self.tagging_categories_input_str_dict.items():
                prompt = self.tagging_input_template.format(input=category_input_str)
                output = self.generate_output(visual_data, prompt)
                outputs[tags_category] = output

        return outputs


    def question_answering(self, visual_data):
        outputs = {}

        if self.mode == 'qa_merged':
            categories_input_str_list = list(self.qa_categories_input_str_dict.values())
            categories_input_str = '\n'.join(categories_input_str_list)
            prompt = self.qa_input_template.format(input=categories_input_str)
            output = self.generate_output(visual_data, prompt)
            outputs['predictions'] = output
        else:
            for question_category, category_input_str in self.qa_categories_input_str_dict.items():
                prompt = self.qa_input_template.format(input=category_input_str)
                output = self.generate_output(visual_data, prompt)
                outputs[question_category] = output

        return outputs


    def chat(self, visual_data):
        outputs = {}
        qa_template = self.qa_input_template + ' {}.'
        context = ''

        for question in self.questions:
            if question is None:
                continue

            prompt = context + self.qa_input_template.format(question)
            output = self.generate_output(visual_data, prompt)

            question_anwer = qa_template.format(question, output)
            context += question_anwer + ' '
            
            outputs[question] = output

        return outputs


    def outputs_to_string(self, outputs):
        output_strs = []
        for k, v in outputs.items():
            output_str = self.output_template.format(k, v)
            output_strs.append(output_str)
        return output_strs


    def __call__(
        self, 
        visual_data: Union[List[np.ndarray], np.ndarray]
    ):
        if self.mode == 'prompted':
            outputs = self.prompted_captioning(visual_data)
        elif self.mode in ['qa', 'qa_merged']:
            outputs = self.question_answering(visual_data)
        elif self.mode in ['tagging', 'tagging_merged']:
            outputs = self.tagging(visual_data)
        elif self.mode == 'chat':
            outputs = self.chat(visual_data)
        else:
            outputs = self.simple_captioning(visual_data)
            
        return outputs