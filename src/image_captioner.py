import torch

class ImageCaptioner:
    def __init__(
        self,
        model,
        processor,
        questions,
        device='cpu',
        question_template='Question: {} Answer:',
        output_template='Q: {}\nA: {}'
    ):
        self.model = model
        self.processor = processor
        self.device = device 
        self.questions = questions
        self.question_template = question_template
        self.output_template = output_template


    def __call__(self, frame):
        answers = []

        inputs_image = self.processor(
            images=frame, 
            return_tensors="pt"
        ).to(self.device, torch.float16)
        output_template = 'Q: {}\nA: {}'

        for question in self.questions:
            question_prompt = self.question_template.format(question)

            inputs_text = self.processor(
                text=question_prompt, 
                return_tensors="pt"
            ).to(self.device)

            inputs = {**inputs_image, **inputs_text}

            generated_ids = self.model.generate(
                **inputs, 
                max_length=100,
            )
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            answer = output_template.format(question, generated_text)
            answers.append(answer)

        return answers