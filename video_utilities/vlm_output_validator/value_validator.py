from typing import (
    Union,
    Optional,
    List,
    Dict
)


def validate_value(
    value, 
    expected_output_type: str, 
    validation_params: Optional[Dict]
):
    type_map = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict
    }

    expected_type = type_map.get(expected_output_type.lower())
    if expected_type is None:
        raise ValueError(f"Unsupported type: {expected_output_type}")

    if not isinstance(value, expected_type):
        return None
    
    if validation_params is None:
        return value

    validation_type = validation_params.get('validation_type')

    if validation_type == 'numeric':
        min_value = validation_params.get('min_value', float('-inf'))
        max_value = validation_params.get('max_value', float('inf'))
        
        if not (min_value <= value <= max_value):
            return None

    elif validation_type == 'oneof':
        possible_values = validation_params.get('possible_values', [])
        
        if value not in possible_values:
            return None

    elif validation_type == 'string_length':
        min_length = validation_params.get('min_length', 0)
        max_length = validation_params.get('max_length', float('inf'))
        
        if not (min_length <= len(value) <= max_length):
            return None

    elif validation_type == 'regex':
        import re
        pattern = validation_params.get('pattern', '')
        
        if not re.match(pattern, value):
            return None

    elif validation_type == 'custom':
        custom_func = validation_params.get('custom_func')
        
        if custom_func and callable(custom_func):
            is_valid = custom_func(value)
            if not is_valid:
                return None

    return value