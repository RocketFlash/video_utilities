import re
from typing import (
    Union,
    Optional,
    List,
    Dict
)


class FrameOutputProcessor:
    patterns = {
        "int": r'\b(\d+)\b',
        "float": r'\b(\d+(?:\.\d+)?)\b',
        "list[int]": r'\[\s*(-?\d+\s*(?:,\s*-?\d+\s*)*)\]',
        "list[float]": r'\[\s*(-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*)*)\]',
        "list[str]": r'\[(.*?)\]'
    }
    def __init__(
        self,
        list_delimiter: str = ','
    ):
        self.list_delimiter = list_delimiter

    def __call__(
        self,
        input_string: str,
        expected_output_type: str
    ) -> Optional[Union[int, float, List[int], List[float], List[str], str]]:

        input_string = input_string.strip()

        if input_string=='None':
            return None

        if expected_output_type == "str":
            return input_string

        n_words = len(input_string.split())

        if n_words == 1:
            str_value = input_string.strip('[],.\'\"')
            if str_value=='None':
                return None
        else:
            pattern = self.patterns.get(expected_output_type)
            if not pattern:
                print(f"Unsupported expected output type: {expected_output_type}")
                return None

            match_curr = re.search(pattern, input_string)
            if not match_curr:
                print(f"No match found for expected output type {expected_output_type} in string: {input_string}")
                return None
            str_value = match_curr.group(1)

        try:
            if expected_output_type == "int":
                return int(str_value)
            elif expected_output_type == "float":
                return float(str_value)
            elif expected_output_type in ["list[int]", "list[float]", "list[str]"]:
                items = str_value.split(self.list_delimiter)
                items = [item.strip('[],.\'\"') for item in items]

                if expected_output_type == "list[int]":
                    return [int(item.strip()) for item in items]
                elif expected_output_type == "list[float]":
                    return [float(item.strip()) for item in items]
                else:
                    return [item.strip() for item in items]
        except ValueError as e:
            print(f"Parsing error: {e}")
            return None