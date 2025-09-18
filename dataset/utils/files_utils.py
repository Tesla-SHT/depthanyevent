import pathlib
import os
import json
from typing import Tuple, Dict, Any

def get_dsec_dev_full_path()-> str:
    this_file_absolute_path = pathlib.Path(__file__).parent.resolve()
    elements = str(this_file_absolute_path).split("/")
    assert "nut_events" in elements
    position = elements.index("nut_events")
    position_from_the_end = len(elements) - position
    tmp_path = this_file_absolute_path
    for i in range(position_from_the_end):
        tmp_path = tmp_path.parent.resolve()
    return tmp_path

def get_configuration(configuration_name)-> Tuple[Dict[str, Any], str]:
    dsec_dev_absolute_path = get_dsec_dev_full_path()
    if configuration_name[-4:] != ".json":
        configuration_name += ".json"
    configuration_file_path = os.path.join(dsec_dev_absolute_path, "configs", configuration_name)
    with open(configuration_file_path, "r") as configuration_file:
        configuration = json.load(configuration_file)
    return configuration, configuration_file_path

def format_int_with_zeros(my_number, chars_number):
    my_number = str(my_number)
    if len(my_number) > chars_number:
        raise Exception(f"Trying to format {my_number} with {chars_number} character(s)! :-(")
    while len(my_number) < chars_number:
        my_number = "0" + my_number
    return my_number