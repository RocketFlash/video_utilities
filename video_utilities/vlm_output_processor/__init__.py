import re
from typing import (
    Union,
    Optional,
    List,
    Dict
)


class VLMOutputProcessor:
    patterns = {
        "int": r'-?\d+',
        "float": r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',
        "list[int]":  r'\[(?:\s*-?\d+\s*(?:,\s*-?\d+\s*)*)\]',
        "list[float]": r'\[(?:\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*)*)\]',
        "list[str]": r'\[((?:[^\[\],]+|\'[^\']*\'|"[^"]*")(?:\s*,\s*(?:[^\[\],]+|\'[^\']*\'|"[^"]*"))*)\]',
        "dict[list[str]]": r'\{((?:\s*[\'"]?\w+[\'"]?\s*:\s*\[((?:[^\[\],]+|\'[^\']*\'|"[^"]*")(?:\s*,\s*(?:[^\[\],]+|\'[^\']*\'|"[^"]*"))*)\]\s*,?\s*)+)\}'
    }

    def __call__(
        self,
        input_string: str,
        expected_output_type: str
    ):

        input_string = input_string.strip()

        if input_string in ['None', 'none']:
            return None

        if expected_output_type == "str":
            return input_string

        n_words = len(input_string.split())

        if n_words == 1:
            str_value = input_string.strip('[],.\'\"')
            if str_value in ['None', 'none']:
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
            str_value = match_curr.group(0)

        try:
            if expected_output_type == "int":
                return int(str_value)
            elif expected_output_type == "float":
                return float(str_value)
            elif expected_output_type == "dict[list[str]]":
                pairs = re.findall(r'([\'"]?\w+[\'"]?)\s*:\s*\[((?:[^\[\],]+|\'[^\']*\'|"[^"]*")(?:\s*,\s*(?:[^\[\],]+|\'[^\']*\'|"[^"]*"))*)\]', str_value)
                result = {}
                for key, value in pairs:
                    key = key.strip("'\"")
                    items = re.findall(r'\'([^\']*?)\'|"([^"]*?)"|([^\s,\[\]]+)', value)
                    result[key] = [item[0] or item[1] or item[2] for item in items]
                return result
            elif expected_output_type in ["list[int]", "list[float]", "list[str]"]:
                if expected_output_type == "list[int]":
                    items = re.findall(r'-?\d+', str_value)
                    return [int(item.strip()) for item in items]
                elif expected_output_type == "list[float]":
                    items = re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?', str_value)
                    return [float(item) for item in items]
                else:
                    items = re.findall(r'\'([^\']*?)\'|"([^"]*?)"|([^\s,\[\]]+)', str_value)
                    return [item[0] or item[1] or item[2] for item in items]
        except ValueError as e:
            print(f"Parsing error: {e}")
            return None