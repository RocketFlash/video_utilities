import cv2
import json
import numpy as np
from urllib.request import urlopen


def read_image_npy(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_value_if_not_none(
    dict_data, 
    dict_key, 
    default_value=None
):
    if dict_data is not None:
        if dict_key in dict_data:
            dict_value = dict_data[dict_key]
        else:
            dict_value = default_value
    else:
        dict_value = None
        
    return dict_value


def read_prompt_template(file_path):
    with open(file_path, encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template


def save_prompt_template(prompt_template, save_path):
    with open(save_path, "w") as outfile:
        outfile.write(prompt_template)


def read_queries_dict(file_path):
    with open(file_path) as f:
        queries_dict = json.load(f)
    return queries_dict


def save_queries_dict(queries_dict, save_path):
    with open(save_path, "w") as outfile: 
        json.dump(queries_dict, outfile, indent=4)


def generate_instruction_str(category_dict):
    instruction_str = ''
    validation_str = ''
    
    if 'validation_params' in category_dict:
        validation_type = category_dict['validation_params']['validation_type']
        if validation_type == 'numeric':
            min_value = category_dict['validation_params']['min_value']
            max_value = category_dict['validation_params']['max_value']
            validation_str += f'between {min_value} and {max_value}'
        elif validation_type == 'oneof':
            one_of_values = category_dict['validation_params']['possible_values']
            one_of_str = ', '.join(one_of_values)
            validation_str += f', and can be only one of the following values: [{one_of_str}]'

    expected_output_type = category_dict['expected_output_type']
    if expected_output_type=='list':
        expected_element_type = category_dict['expected_element_type']
        instruction_str += 'The answer should be in the form of a list of values, where each element is '
        if expected_element_type in ['int', 'float']:
            expected_element_type_str = f'an {expected_element_type} number '
        else:
            expected_element_type_str = 'a string '
        instruction_str += expected_element_type_str
    else:
        instruction_str += 'The answer should be '
        if expected_output_type in ['int', 'float']:
            expected_output_type_str = f'an {expected_output_type} number '
        else:
            expected_output_type_str = 'a string '
        instruction_str += expected_output_type_str

    instruction_str += validation_str
    return instruction_str