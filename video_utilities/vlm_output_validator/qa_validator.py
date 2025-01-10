from typing import (
    Union,
    Optional,
    List,
    Dict
)
from .value_validator import validate_value


class QAOutputValidator:
    def __init__(
        self, 
        questions_dict: Dict[str, Dict] = {},
    ):
        self.questions_dict = questions_dict

    def __call__(
        self,
        output_raw
    ):
        if isinstance(self.questions_dict, dict):
            output_validated = {
                q_cat: None for q_cat in self.questions_dict.keys()
            }
            for question_cat_name, question_cat_info in self.questions_dict.items():
                expected_output_type = question_cat_info['expected_output_type']
                validation_params = question_cat_info.get('validation_params', None)
                if output_raw is not None:
                    if question_cat_name in output_raw:
                        value = output_raw[question_cat_name]
                        value_validated = validate_value(
                            value, 
                            expected_output_type=expected_output_type,
                            validation_params=validation_params
                        )
                        output_validated[question_cat_name] = value_validated
            return output_validated
        else:
            return None