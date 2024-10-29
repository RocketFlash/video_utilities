import torch
from typing import (
    Union,
    Optional,
    List,
    Dict
)


class FrameCaptioner:
    def __init__(
        self,
        model_name: str = 'Salesforce/blip2-opt-2.7b',
        questions: List = [None],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float16,
        generation_params: Dict = {},
        prompt: Union[str, List] = 'In this video frame',
        tags: Union[Dict, List] = [],
        tags_desc: Optional[Union[Dict, str]] = None,
        qa_input_template: str = 'Question: {} Answer:',
        qa_output_template: Optional[str] = None,
        tagging_input_template: str = 'Based on the visual content of the video frame, choose the tags that best describe {} what is shown. Provide the results in the form of a list separated by commas. If no tags apply, state "None". \n\nList of tags: \n{}',
        tagging_output_template: Optional[str] = None,
        mode: str = 'simple', # ['simple', 'prompted', 'qa', 'chat']
        use_quantization: bool = False
    ):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.generation_params = generation_params
        self.device = device
        self.dtype = dtype

        model, processor = self.get_model_and_processor(
            model_name=model_name
        )

        self.model = model
        self.processor = processor
        
        self.set_prompt(prompt)
        self.set_questions(questions)
        self.set_qa_input_template(qa_input_template)
        self.set_qa_output_template(qa_output_template)
        self.set_tags(tags)
        self.set_tags_desc(tags_desc)
        self.set_tagging_input_template(tagging_input_template)
        self.set_tagging_output_template(tagging_output_template)
        self.set_mode(mode)


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

    def set_qa_output_template(self, qa_output_template):
        self.qa_output_template = qa_output_template

    def set_tagging_input_template(self, tagging_input_template):
        self.tagging_input_template = tagging_input_template

    def set_tagging_output_template(self, tagging_output_template):
        self.tagging_output_template = tagging_output_template


    def process_image(self, image):
        inputs_image = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.model.device, self.dtype)
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
        ).to(self.model.device, self.dtype)
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
        return [self.generate_output(image, None)]


    def prompted_frame_captioning(self, image):
        prompts = self.prompt
        if not isinstance(prompts, list):
            prompts = [prompts]

        outputs = []
        for prompt in prompts:
            outputs.append(self.generate_output(image, prompt))

        return outputs
    

    def tagging(self, image):
        tags_dict = self.tags
        tags_desc_dict = self.tags_desc

        if not isinstance(tags_dict, dict):
            tags_dict = {'general': tags_dict}
            if tags_desc_dict is not None:
                tags_desc_dict = {'general': tags_desc_dict}

        outputs = []
        for tags_category, tags_list in tags_dict.items():
            tag_names_str = '\n'.join(tags_list)
            if tags_desc_dict is not None:
                if tags_category in tags_desc_dict:
                    tags_cat_desc = tags_desc_dict[tags_category]
                else:
                    tags_cat_desc = ''
            else:
                tags_cat_desc = ''
            prompt = self.tagging_input_template.format(tags_cat_desc, tag_names_str)
            output = self.generate_output(image, prompt)
            if self.tagging_output_template is not None:
                output = self.tagging_output_template.format(tags_category, output)
            outputs.append(output)

        return outputs


    def question_answering(self, image):
        outputs = []

        for question in self.questions:
            if question is not None:
                prompt = self.qa_input_template.format(question)
            else:
                prompt = None
            output = self.generate_output(image, prompt)

            if self.qa_output_template is not None:
                output = self.qa_output_template.format(question, output)
            outputs.append(output)
        return outputs


    def chat(self, image):
        outputs = []
        qa_template = self.qa_input_template + ' {}.'
        context = ''

        for question in self.questions:
            if question is not None:
                prompt = context + self.qa_input_template.format(question)
            else:
                prompt = None

            answer = self.generate_output(image, prompt)

            if question is not None:
                question_anwer = qa_template.format(question, answer)
                context += question_anwer + ' '
            else:
                context += answer + ' '

            output = self.qa_output_template.format(question, answer)
            outputs.append(output)

        return outputs

    def __call__(self, image):
        if self.mode == 'prompted':
            outputs = self.prompted_frame_captioning(image)
        elif self.mode == 'qa':
            outputs = self.question_answering(image)
        elif self.mode == 'tagging':
            outputs = self.tagging(image)
        elif self.mode == 'chat':
            outputs = self.chat(image)
        else:
            outputs = self.simple_frame_captioning(image)
        return outputs