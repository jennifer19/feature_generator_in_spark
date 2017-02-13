#!/usr/bin/env python
# feature_generator/data_handling/input_args_handler_fg.py

import json
from pkg_resources import resource_string


def json_mapper(category_number, target):
    """ Extract category-specific configurations

    :param category_number      -- specific category
    :param target               -- the targeted characteristic
    :return required_chars      -- the characteristics required for that input set
    """
    # Structure paths to proper category & country json files
    category = category_number.lower()
    local_chars_map_base = '/json_mapping/' + '/'.join([category]) + 'instructions.json'
    local_chars_map = resource_string(__name__, local_chars_map_base)

    # Open json files
    json_decoded_map = json.loads(local_chars_map.decode('utf-8'))
    standard_local_chars = json_decoded_map[target]

    return standard_local_chars

def parse_input_data(json_input_raw):
    """Parse json input and return a dictionary with the relevant ML Pipeline Training
    output requirements
    :param json_input_raw   -- the input in json format
    :return output_dict     -- a dictionary with the parsed input
    """

    # Data source
    json_input_formatted = json.loads(json_input_raw)

    # Parse input JSON to extract file path
    output_dict = dict()
    output_dict['input_file_path'] = json_input_formatted['input_file_path']
    output_dict['base_name'] = json_input_formatted['base_name']
    output_dict['output_file_path'] = json_input_formatted['output_file_path']
    output_dict['independent_var_keyword'] = json_input_formatted['independent_var_keyword'].lower()
    output_dict['dependent_var_keyword'] = json_input_formatted['dependent_var_keyword'].lower()

    return output_dict
