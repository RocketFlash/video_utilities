import torch
from typing import (
    Union,
    Optional,
    List,
    Dict
)
from .config import FrameCaptionerConfig


class FrameCaptioner:
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

        # self.pose_predictor = pose_predictor
        # self.process_frames_only_with_people = process_frames_only_with_people
    
    def get_default_config(
        self,
    ):
        return FrameCaptionerConfig()

    def set_params_from_config(
        self, 
        config: FrameCaptionerConfig
    ):
        for key, value in vars(config).items():
            setattr(self, key, value)
        

    def get_model_and_processor(
        self,
        model_name: str,
    ):
        return None, None

    def set_mode(self, mode):
        self.mode = mode
   
    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_tags(self, tags):
        self.tags = tags

    def set_questions(self, questions):
        self.questions = questions

    def set_qa_input_template(self, qa_input_template):
        self.qa_input_template = qa_input_template

    def set_tagging_input_template(self, tagging_input_template):
        self.tagging_input_template = tagging_input_template

    def set_output_template(self, output_template):
        self.output_template = output_template

    def set_pose_predictor(self, pose_predictor):
        self.pose_predictor = pose_predictor

    # def set_process_frames_only_with_people(self, process_frames_only_with_people):
    #     self.process_frames_only_with_people = process_frames_only_with_people

    def process_image(self, image):
        inputs_image = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        return inputs_image


    def process_text(self, text):
        inputs_text = self.processor(
            text=text,
            return_tensors="pt"
        ).to(self.model.device)
        return inputs_text


    def process_image_and_text(self, image, text):
        inputs = self.processor(
            images=image,
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


    def generate_output(self, image, prompt):
        prompt_len = len(prompt) if prompt is not None else None
        inputs = self.process_image_and_text(
            image=image,
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


    def simple_frame_captioning(self, image):
        output = self.generate_output(image, None)
        outputs = {'caption' : output}
        return outputs


    def prompted_frame_captioning(self, image):
        prompts = self.prompt
        if not isinstance(prompts, list):
            prompts = [prompts]

        outputs = {}
        for prompt in prompts:
            output = self.generate_output(image, prompt)
            outputs[prompt] = output

        return outputs
    

    def tagging(self, image):
        tag_categories_dict = self.tags

        if isinstance(tag_categories_dict, list):
            tag_categories_dict = dict(
                general=dict(
                    tags=tag_categories_dict,
                    description=''
                )
            )

        categories_input_str_dict = {} 
        for tag_category, tag_category_dict in tag_categories_dict.items():
            tag_list = tag_category_dict['tags']
            tags_cat_desc = tag_category_dict['description']
            tag_names_str = ', '.join(tag_list)

            category_input_str = (
                f'category name: {tag_category}\n' 
                f'category description: {tags_cat_desc}\n' 
                f'available tags: [{tag_names_str}]\n'
            )
            categories_input_str_dict[tag_category] = category_input_str
        
        outputs = {}
        if self.mode == 'tagging_merged':
            categories_input_str_list = list(categories_input_str_dict.values())
            categories_input_str = '\n'.join(categories_input_str_list)
            prompt = self.tagging_input_template.format(input=categories_input_str)
            output = self.generate_output(image, prompt)
            outputs['predictions'] = output
        else:
            for tags_category, category_input_str in categories_input_str_dict.items():
                prompt = self.tagging_input_template.format(input=category_input_str)
                output = self.generate_output(image, prompt)
                outputs[tags_category] = output

        return outputs
    

    def question_answering(self, image):
        outputs = {}

        question_categories_dict = self.questions

        if isinstance(question_categories_dict, list):
            questions_list = question_categories_dict
            question_categories_dict = {}

            for i, question in enumerate(questions_list):
                question_category = f'question_{i}'
                question_categories_dict[question_category] = dict(
                    question=question,
                    instruction='',
                    expected_output_type='str',
                    validation_params=None
                )
        
        categories_input_str_dict = {}
        for question_category, question_category_dict in question_categories_dict.items():
            question = question_category_dict['question']
            question_category_desc = question_category_dict['description']
            instruction = question_category_dict['instruction']

            category_input_str = (
                f'category name: {question_category}\n' 
                f'category description: {question_category_desc}\n'
                f'question: {question}\n' 
                f'instruction: {instruction}\n' 
            )
            categories_input_str_dict[question_category] = category_input_str
        
        outputs = {}
        if self.mode == 'qa_merged':
            categories_input_str_list = list(categories_input_str_dict.values())
            categories_input_str = '\n'.join(categories_input_str_list)
            prompt = self.qa_input_template.format(input=categories_input_str)
            output = self.generate_output(image, prompt)
            outputs['predictions'] = output
        else:
            for question_category, category_input_str in categories_input_str_dict.items():
                prompt = self.qa_input_template.format(input=category_input_str)
                output = self.generate_output(image, prompt)
                outputs[question_category] = output

        return outputs


    def chat(self, image):
        outputs = {}
        qa_template = self.qa_input_template + ' {}.'
        context = ''

        for question in self.questions:
            if question is None:
                continue

            prompt = context + self.qa_input_template.format(question)
            output = self.generate_output(image, prompt)

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


    def __call__(self, image):
        # if self.process_frames_only_with_people:
        #     is_frame_with_people = False
        #     if self.pose_predictor is not None:
        #         result_pose_estimation = self.pose_predictor([image])[0]
        #         landmarks_2d = result_pose_estimation.landmarks_2d
        #         if landmarks_2d is not None:
        #             if len(landmarks_2d) > 0:
        #                 is_frame_with_people = True

        #     if not is_frame_with_people:
        #         if 'merged' in self.mode:
        #             outputs = dict(predictions='')
        #         else:
        #             outputs = dict()
        #         return outputs

        if self.mode == 'prompted':
            outputs = self.prompted_frame_captioning(image)
        elif self.mode in ['qa', 'qa_merged']:
            outputs = self.question_answering(image)
        elif self.mode in ['tagging', 'tagging_merged']:
            outputs = self.tagging(image)
        elif self.mode == 'chat':
            outputs = self.chat(image)
        else:
            outputs = self.simple_frame_captioning(image)
        return outputs