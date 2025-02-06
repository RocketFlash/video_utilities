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