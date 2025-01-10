from typing import (
    Union,
    Optional,
    List,
    Dict
)


class TaggingOutputValidator:
    def __init__(
        self, 
        tags_dict: Dict[str, Dict] = {},
    ):
        self.tags_dict = tags_dict

    def __call__(
        self,
        output_raw
    ):
        if isinstance(self.tags_dict, dict):
            output_validated = {tag_cat_name: None for tag_cat_name in self.tags_dict.keys()}
            for tag_cat_name, tag_cat_info in self.tags_dict.items():
                available_tags = tag_cat_info['tags']
                if output_raw is not None:
                    if tag_cat_name in output_raw:
                        correct_tags = []
                        for tag in output_raw[tag_cat_name]:
                            if tag in available_tags:
                                correct_tags.append(tag)
                        if correct_tags:
                            output_validated[tag_cat_name] = correct_tags
            return output_validated
        else:
            return None