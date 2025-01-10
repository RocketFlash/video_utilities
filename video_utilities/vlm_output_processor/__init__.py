import re
from typing import (
    Union,
    Optional,
    List,
    Dict
)


def parse_value(value):
    if value == 'None':
        return None
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    elif '.' in value or 'e' in value.lower():
        try:
            return float(value)
        except ValueError:
            return value
    else:
        try:
            return int(value)
        except ValueError:
            return value


class VLMOutputProcessor:
    patterns = {
        "int": r'-?\d+',
        "float": r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',
        "str": r'\'([^\']*?)\'|"([^"]*?)"|([^\s,\[\]]+)',
        "list[int]":  r'\[(?:\s*-?\d+\s*(?:,\s*-?\d+\s*)*)\]',
        "list[float]": r'\[(?:\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*)*)\]',
        "list[str]": r'\[((?:[^\[\],]+|\'[^\']*\'|"[^"]*")(?:\s*,\s*(?:[^\[\],]+|\'[^\']*\'|"[^"]*"))*)\]',
        "dict": r'\{(?:\s*["\'](\w+)["\']\s*:\s*((?:None|True|False|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|"[^"]*"|\'[^\']*\'))\s*,?\s*)*\}',
        "dict[list[str]]" : r'\{((?:\s*[\'"]?\w+[\'"]?\s*:\s*(?:\[((?:[^\[\],]+|\'[^\']*\'|"[^"]*")(?:\s*,\s*(?:[^\[\],]+|\'[^\']*\'|"[^"]*"))*)\]|None)\s*,?\s*)+)\}'
    }

    def __init__(self, verbose=False):
        self.verbose = verbose

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
                if self.verbose:
                    print(f"Unsupported expected output type: {expected_output_type}")
                return None

            match_curr = re.search(pattern, input_string)
            if not match_curr:
                if self.verbose:
                    print(f"No match found for expected output type {expected_output_type} in string: {input_string}")
                return None
            str_value = match_curr.group(0)

        try:
            if expected_output_type == "int":
                return int(str_value)
            elif expected_output_type == "float":
                return float(str_value)
            elif expected_output_type == "dict[list[str]]":
                pairs = re.findall(r'([\'"]?\w+[\'"]?)\s*:\s*((?:\[((?:[^\[\],]+|\'[^\']*\'|"[^"]*")(?:\s*,\s*(?:[^\[\],]+|\'[^\']*\'|"[^"]*"))*)\]|None))', str_value)
                result = {}
                for key, value, _ in pairs:
                    if value.strip() == 'None':
                        result[key] = None
                    else:
                        key = key.strip("'\"")
                        items = re.findall(self.patterns['str'], value)
                        result[key] = [item[0] or item[1] or item[2] for item in items]
                return result
            elif expected_output_type == 'dict':
                pairs = re.findall(r'["\'](\w+)["\']\s*:\s*((?:None|True|False|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|"[^"]*"|\'[^\']*\'))', str_value)
                return {key: parse_value(value) for key, value in pairs}
            elif expected_output_type in ["list[int]", "list[float]", "list[str]"]:
                if expected_output_type == "list[int]":
                    items = re.findall(self.patterns['int'], str_value)
                    return [int(item.strip()) for item in items]
                elif expected_output_type == "list[float]":
                    items = re.findall(self.patterns['float'], str_value)
                    return [float(item) for item in items]
                else:
                    items = re.findall(self.patterns['str'], str_value)
                    return [item[0] or item[1] or item[2] for item in items]
        except ValueError as e:
            if self.verbose:
                print(f"Parsing error: {e}")
            return None