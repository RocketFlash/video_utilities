from typing import (
    Union,
    Optional,
    List,
    Dict
)


class TaggingOutputValidator:
    def __init__(
        self, 
        tags: Union[Dict[str, Dict[str, str]], List[str]] = [],
    ):
        self.tags = tags

    def __call__(
        self,
        output_raw
    ):
        if isinstance(self.tags, dict):
            output_validated = {tag_cat: None for tag_cat in self.tags.keys()}
            for tag_cat, tag_cat_info in self.tags.items():
                available_tags = tag_cat_info['tags']
                if tag_cat in output_raw:
                    correct_tags = []
                    for tag in output_raw[tag_cat]:
                        if tag in available_tags:
                            correct_tags.append(tag)
                    if correct_tags:
                        output_validated[tag_cat] = correct_tags
            return output_validated
        else:
            return None