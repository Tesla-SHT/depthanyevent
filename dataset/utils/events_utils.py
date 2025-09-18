import pathlib
import os
from typing import Dict, Any


from .files_utils import get_configuration
from ..events.events_representations import KNOWN_REPRESENTATIONS
from ..events.events_representations import EventRepresentation

class RepresentationsLoader:
    def __init__(self):
        self.representations = self.get_and_check_model_representation_file()

    def load(self, representation_name)-> EventRepresentation:
        if representation_name in self.representations.keys():
            configuration = self.representations[representation_name]
        else:
            raise Exception(f"Representation '{representation_name}' not found.\n"
                            f"Found representations: {list(self.representations.keys())}")

        if configuration["representation_type"] in KNOWN_REPRESENTATIONS.keys():
            return KNOWN_REPRESENTATIONS[configuration["representation_type"]].from_configuration(configuration)
        else:
            raise Exception(f"Impossible to get here! :-)")

    @staticmethod
    def get_and_check_model_representation_file() -> Dict[str, Any]:
        representations, representations_file_path = get_configuration("representations")
        if not isinstance(representations, Dict):
            raise Exception(f"Error in {representations_file_path}.\n"
                            f"I was expecting a Dictionary but it's not! :-(")
        for representation_name, representation in representations.items():
            if not isinstance(representation, Dict):
                raise Exception(f"Error in {representations_file_path}.\n"
                                f"\"{representation_name}\" representation should be a Dictionary.")
            if "representation_type" not in representation.keys():
                raise Exception(f"Error in {representations_file_path}.\n"
                                f"\"{representation_name}\" representation does not contain a"
                                f" 'representation_type' key.")
            if representation["representation_type"] not in KNOWN_REPRESENTATIONS.keys():
                raise Exception(f"Error in {representations_file_path}.\n"
                                f"\"{representation['representation_type']}\" of \"{representation_name}\" "
                                f"does not match any known representation.\n"
                                f"Known Representation: {list(KNOWN_REPRESENTATIONS.keys())}")
        return representations

