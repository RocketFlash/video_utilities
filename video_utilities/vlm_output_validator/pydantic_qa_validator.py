import re
from typing import (
    Dict, 
    List,
    Any, 
    Optional, 
    Union, 
    Literal,
    Type
)

try:
    from pydantic import (
    BaseModel, 
    Field, 
    create_model,
    validator
)
    PYDANTIC_INSTALLED = True
except ImportError:
    PYDANTIC_INSTALLED = False


class PydanticQAOutputValidator:
    type_map = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict
    }
    
    def __init__(
        self, 
        questions_dict: Dict[str, Any],
        verbose: bool = False
    ):
        self.questions_dict = questions_dict
        self.verbose = verbose
        self.validation_models = self._create_validation_models()

    def _create_validation_models(self) -> Dict[str, Type[BaseModel]]:
        """Create Pydantic models for each question category"""
        models = {}
        for category, config in self.questions_dict.items():
            if config["expected_output_type"] == "list":
                # Create a model for list elements
                models[category] = self._create_element_model(config)
            else:
                # Create a model for single values
                field_type = self._get_field_type(config)
                field_kwargs = self._get_field_kwargs(config)
                validators = self._get_validators(config)

                models[category] = create_model(
                    f"{category.capitalize()}Validator",
                    value=(Optional[field_type], Field(**field_kwargs)),
                    __validators__=validators,
                    __base__=BaseModel
                )
        return models

    def _create_element_model(self, config: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic model for list elements"""
        field_type = self._get_field_type(config)
        field_kwargs = self._get_field_kwargs(config)
        validators = self._get_validators(config)

        return create_model(
            f"{config['expected_element_type'].capitalize()}ElementValidator",
            value=(Optional[field_type], Field(**field_kwargs)),
            __validators__=validators,
            __base__=BaseModel
        )
    
    def _map_type(self, type_str: str, config: Dict[str, Any]) -> type:
        """Map string type to Python type, including oneof validation"""
        if "validation_params" in config:
            if config["validation_params"]["validation_type"] == "oneof":
                return Literal[tuple(config["validation_params"]["possible_values"])]
        return self.type_map.get(type_str, str)


    def _get_field_type(self, config: Dict[str, Any]) -> type:
        """Determine the appropriate type for Pydantic field"""
        if config["expected_output_type"] == "list":
            return self._map_type(config["expected_element_type"], config)
        return self._map_type(config["expected_output_type"], config)


    def _get_field_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get validation parameters for Pydantic field"""
        kwargs = {"default": None}
        
        if "validation_params" in config:
            params = config["validation_params"]
        
            if params["validation_type"] == "numeric":
                kwargs.update({
                    "ge": params.get("min_value", float("-inf")),
                    "le": params.get("max_value", float("inf"))
                })
            elif params["validation_type"] == "string_length":
                kwargs.update({
                    "min_length": params.get("min_length", 0),
                    "max_length": params.get("max_length", float("inf"))
                })
        return kwargs


    def _get_validators(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add custom validators based on validation_type"""
        
        validators = {}
        if "validation_params" in config:
            params = config["validation_params"]
            if params["validation_type"] == "regex":
                pattern = re.compile(params["pattern"])

                @validator("value", allow_reuse=True)
                def validate_regex(cls, v):
                    if v is not None and not pattern.match(v):
                        raise ValueError(f"Value must match regex pattern: {params['pattern']}")
                    return v
                validators["validate_regex"] = validate_regex

            elif params["validation_type"] == "custom":
                @validator("value", allow_reuse=True)
                def validate_custom(cls, v):
                    if v is not None and not self._custom_validation(v, params):
                        raise ValueError("Custom validation failed")
                    return v
                validators["validate_custom"] = validate_custom

        return validators


    def _custom_validation(self, value: Any, params: Dict[str, Any]) -> bool:
        """Placeholder for custom validation logic"""
        return True


    def _validate_list(self, category: str, raw_list: List[Any]) -> Optional[List[Any]]:
        """Validate each element in a list and filter out invalid elements"""
        if not isinstance(raw_list, list):
            return None

        validated_elements = []
        model = self.validation_models[category]

        for element in raw_list:
            try:
                # Validate each element
                instance = model(value=element)
                if instance.value is not None:
                    validated_elements.append(instance.value)
            except:
                if self.verbose:
                    print(f"Validation failed for list element in {category}")

        return validated_elements if validated_elements else None


    def __call__(self, parsed_output: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Validate and normalize the parsed output"""
        if not isinstance(parsed_output, dict):
            return None

        validated_result = {}
        for category in self.questions_dict:
            raw_value = parsed_output.get(category)

            if self.questions_dict[category]["expected_output_type"] == "list":
                validated_result[category] = self._validate_list(category, raw_value)
            else:
                model = self.validation_models[category]
                try:
                    instance = model(value=raw_value)
                    validated_result[category] = instance.value
                except:
                    if self.verbose:
                        print(f"Validation failed for {category}")
                    validated_result[category] = None

        return validated_result