from openai import OpenAI
import numpy as np
from PIL import Image
from .frame_captioner import FrameCaptioner


class DeepSeekAPIFrameCaptioner(FrameCaptioner):

    def get_model_and_processor(
        self,
        model_name: str,
    ):
        processor = None
        model = OpenAI(
            api_key=self.additional_params['api_key'], 
            base_url=self.additional_params['base_url']
        )

        return model, processor

    def generate_output(
        self, 
        image, 
        prompt
    ):
        pass

        return 