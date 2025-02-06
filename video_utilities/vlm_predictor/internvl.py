import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel

)
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from ..utils import get_value_if_not_none
from .vlm_predictor import VLMPredictor


def find_closest_aspect_ratio(
    aspect_ratio, 
    target_ratios, 
    width, 
    height, 
    image_size
):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, 
    min_num=1, 
    max_num=12, 
    image_size=448, 
    use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class InternVL_VLMPredictor(VLMPredictor):
    def get_model_and_processor(
        self,
        model_name: str,
    ):
        
        use_flash_attn = get_value_if_not_none(
            dict_data=self.additional_params, 
            dict_key='use_flash_attn', 
            default_value=True
        )

        load_in_8bit = get_value_if_not_none(
            dict_data=self.additional_params, 
            dict_key='load_in_8bit', 
            default_value=True
        )

        low_cpu_mem_usage = get_value_if_not_none(
            dict_data=self.additional_params, 
            dict_key='low_cpu_mem_usage', 
            default_value=True
        )

        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            use_fast=False
        )
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            use_flash_attn=use_flash_attn,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=self.device,
            trust_remote_code=True
        ).eval()
        self.tokenizer = tokenizer

        self.input_size = get_value_if_not_none(
            dict_data=self.additional_params, 
            dict_key='input_size', 
            default_value=448
        )

        self.max_num = get_value_if_not_none(
            dict_data=self.additional_params, 
            dict_key='max_num', 
            default_value=12
        )

        self.preprocessing_transform = self.build_transform(self.input_size)

        return model, processor


    def build_transform(
        self,
        input_size
    ):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


    def preprocess_image(
        self,
        image, 
    ):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert('RGB')
        images = dynamic_preprocess(
            image, 
            image_size=self.input_size, 
            use_thumbnail=True, 
            max_num=self.max_num
        )
        pixel_values = [self.preprocessing_transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values.to(torch.bfloat16).to(self.device)

    def process_image_and_text(self, image, text):
        pixel_values = self.preprocess_image(image)

        response, history = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            text, 
            generation_config=self.generation_params, 
            history=None, 
            return_history=True
        )

        return response

    def generate_output(self, image, prompt):
        if prompt is None:
            prompt = ''

        generated_text = self.process_image_and_text(
            image=image,
            text=prompt
        )
        return generated_text