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
        config: Optional[FrameCaptionerConfig] = None,
        model_name: str = 'Salesforce/blip2-opt-2.7b',
        questions: List[str] = ['What is this?'],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float16,
        generation_params: Dict = {},
        prompt: Union[str, List] = 'In this video frame',
        tags: Union[Dict[str, Dict[str, str]], List[str]] = [],
        qa_input_template: str = 'Question: {} Answer:',
        tagging_input_template: str = 'Based on the visual content of the video frame, choose the tags that best describe {} what is shown. Provide the results in the form of a list separated by commas. If no tags apply, state "None". \n\nList of tags: \n{}',
        output_template: str = '{} : {}',
        mode: str = 'simple', # ['simple', 'prompted', 'tagging', 'qa', 'chat', 'tagging_merged']
        use_quantization: bool = False,
        attn_implementation: str = 'sdpa'
    ):
        if config is None:
            config = FrameCaptionerConfig(
                model_name=model_name,
                device=device,
                dtype=dtype,
                questions=questions,
                tags=tags,
                prompt=prompt,
                mode=mode,
                use_quantization=use_quantization,
                generation_params=generation_params,
                qa_input_template=qa_input_template,
                tagging_input_template=tagging_input_template,
                output_template=output_template,
                attn_implementation=attn_implementation
            )
        self.config = config
        self.set_params_from_config(config)
        

    def set_params_from_config(
        self, 
        config: FrameCaptionerConfig
    ):
        self.model_name = config.model_name
        self.use_quantization = config.use_quantization
        self.generation_params = config.generation_params
        self.device = config.device
        self.dtype = config.dtype
        self.attn_implementation = config.attn_implementation

        model, processor = self.get_model_and_processor(
            model_name=config.model_name
        )

        self.model = model
        self.processor = processor
        
        self.set_prompt(config.prompt)
        self.set_questions(config.questions)
        self.set_qa_input_template(config.qa_input_template)
        self.set_tags(config.tags)
        self.set_tags_desc(config.tags_desc)
        self.set_tagging_input_template(config.tagging_input_template)
        self.set_output_template(config.output_template)
        self.set_mode(config.mode)


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

    def set_tags_desc(self, tags_desc):
        self.tags_desc = tags_desc

    def set_questions(self, questions):
        self.questions = questions

    def set_qa_input_template(self, qa_input_template):
        self.qa_input_template = qa_input_template

    def set_tagging_input_template(self, tagging_input_template):
        self.tagging_input_template = tagging_input_template

    def set_output_template(self, output_template):
        self.output_template = output_template


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
                    desc=''
                )
            )

        categories_input_str_dict = {} 
        for tag_category, tag_category_dict in tag_categories_dict.items():
            tag_list = tag_category_dict['tags']
            tags_cat_desc = tag_category_dict['desc']
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

        for question in self.questions:
            if question is None:
                continue

            prompt = self.qa_input_template.format(question)
            output = self.generate_output(image, prompt)
            outputs[question] = output

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
        if self.mode == 'prompted':
            outputs = self.prompted_frame_captioning(image)
        elif self.mode == 'qa':
            outputs = self.question_answering(image)
        elif self.mode in ['tagging', 'tagging_merged']:
            outputs = self.tagging(image)
        elif self.mode == 'chat':
            outputs = self.chat(image)
        else:
            outputs = self.simple_frame_captioning(image)
        return outputs