import torch
from typing import (
    Union,
    Optional,
    List,
    Dict
)


class ImageCaptioner:
    def __init__(
        self,
        model_name: str = 'Salesforce/blip2-opt-2.7b',
        questions: List = [None],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float16,
        generation_params: Dict = {},
        prompt: str = 'In this video frame',
        question_template: str = 'Question: {} Answer:',
        output_template: str = 'Q: {}\nA: {}',
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
        self.prompt = prompt
        self.questions = questions
        self.question_template = question_template
        self.output_template = output_template
        self.mode = mode


    def get_model_and_processor(
        self,
        model_name: str,
    ):
        return None, None

    def set_mode(self, mode):
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_questions(self, questions):
        self.questions = questions

    def set_question_template(self, question_template):
        self.question_template = question_template

    def set_output_template(self, output_template):
        self.output_template = output_template


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
        prompt_len=None
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


    def simple_image_captioning(self, image):
        return [self.generate_output(image, None)]


    def prompted_image_captioning(self, image):
        return [self.generate_output(image, self.prompt)]


    def question_answering(self, image):
        outputs = []

        for question in self.questions:
            if question is not None:
                prompt = self.question_template.format(question)
            else:
                prompt = None
            answer = self.generate_output(image, prompt)
            output = self.output_template.format(question, answer)
            outputs.append(output)
        return outputs


    def chat(self, image):
        outputs = []
        qa_template = self.question_template + ' {}.'
        context = ''

        for question in self.questions:
            if question is not None:
                prompt = context + self.question_template.format(question)
            else:
                prompt = None

            answer = self.generate_output(image, prompt)

            if question is not None:
                question_anwer = qa_template.format(question, answer)
                context += question_anwer + ' '
            else:
                context += answer + ' '

            output = self.output_template.format(question, answer)
            outputs.append(output)

        return outputs

    def __call__(self, image):
        if self.mode == 'prompted':
            outputs = self.prompted_image_captioning(image)
        elif self.mode == 'qa':
            outputs = self.question_answering(image)
        elif self.mode == 'chat':
            outputs = self.chat(image)
        else:
            outputs = self.simple_image_captioning(image)
        return outputs