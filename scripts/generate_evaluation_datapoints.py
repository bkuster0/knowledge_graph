import os
import re
from dataclasses import dataclass
from typing import List, Dict
import json


@dataclass
class EvaluationDatapoint:
    full_image_path: str = '/knowledge_graph/example_single_node_dataset/smoke_detector/0.jpg'
    full_metadata_path: str = '/knowledge_graph/example_single_node_dataset/smoke_detector/0_meta.json'
    number_of_remaining_steps: int = 1
    is_final_step: int = 0

    def to_dict(self):
        return vars(self)


@dataclass
class RegexMatch:
    match_idx_str: str = '0000'
    full_filename: str = '0000_qa.json'


def convert_dataset_to_evaluation_datapoints(folder, output_json_full_path = None) -> List[EvaluationDatapoint]:
    """
    Read a dataset or single node (walks all folders within folder) to output evaluation datapoints.
    (Finds matches between images and metadata and packs them into EvaluationDatapoint class).

    Example use:
    >>> eval_datapoints = convert_dataset_to_evaluation_datapoints(folder = '/knowledge_graph/example_single_node_dataset/', output_json_full_path = '/knowledge_graph/training_data_jsons/data.json')

    Args:
    folder: Folder within which all folders will be walked (os.walk)

    Returns: List[EvaluationDatapoint], e.g.

    [EvaluationDatapoint(full_image_path='/knowledge_graph/example_single_node_dataset/0.png',
                         full_metadata_path='/knowledge_graph/example_single_node_dataset/0_meta.json',
                         number_of_remaining_steps=1,
                         is_final_step=0)]
    """

    IMAGE_EXTENSION_REGEX = '\.(png|jpg|jpeg)'
    METADATA_EXTENSION_REGEX = '_meta\.json'

    image_extension_re = re.compile(IMAGE_EXTENSION_REGEX)
    metadata_extension_re = re.compile(METADATA_EXTENSION_REGEX)

    evaluation_datapoints = []
    for (root,dirs,files) in os.walk(folder,topdown=True):
        print(root)

        # TODO: Ignore hidden folders
        is_hidden_folder = False
        if is_hidden_folder:
            continue

        # Find all image files
        image_regex_matches = filter_files_by_regex_extension(extension_regex_pattern = image_extension_re,
                                                              list_of_strings = files)
        image_idx_to_regex_match_dict = {i.match_idx_str : i for i in image_regex_matches}

        # Find all metadata files
        metadata_regex_matches = filter_files_by_regex_extension(extension_regex_pattern = metadata_extension_re,
                                                              list_of_strings = files)
        metadata_idx_to_regex_match_dict = {i.match_idx_str : i for i in metadata_regex_matches}

        for idx_str, image_regex_match in image_idx_to_regex_match_dict.items():
            # Try to match Image idx_str to metadata idx_str
            if idx_str not in metadata_idx_to_regex_match_dict.keys():
                continue # No match

            metadata_regex_match = metadata_idx_to_regex_match_dict[idx_str]
            full_image_path = os.path.join(root, image_regex_match.full_filename)
            full_metadata_path = os.path.join(root, metadata_regex_match.full_filename)

            evaluation_datapoint = EvaluationDatapoint(full_image_path=full_image_path, full_metadata_path = full_metadata_path)

            # Read metadata
            with open(full_metadata_path, 'r') as metadata_file:
                meta_json = json.load(metadata_file)
                #print(meta_json)
                evaluation_datapoint.number_of_remaining_steps = meta_json["number_of_remaining_steps"]
                evaluation_datapoint.is_final_step = meta_json["is_final_step"]

            evaluation_datapoints.append(evaluation_datapoint)

    # Save as JSON
    if output_json_full_path is not None:
        datapoints_dict = [datapoint.to_dict() for datapoint in evaluation_datapoints]
        json_data = json.dumps(datapoints_dict, indent = 4)
        with open(output_json_full_path, 'w') as json_file:
            json_file.write(json_data)

    return evaluation_datapoints


def filter_files_by_regex_extension(extension_regex_pattern: re.Pattern = re.compile('\.(png|jpg|jpeg)'),
                                    list_of_strings: List[str] = ['0.jpg']) -> List[RegexMatch]:
    list_of_regex_matches = []

    for file in list_of_strings:
        match = extension_regex_pattern.search(file)
        if match is None:
            continue
        idx = file[:match.pos+1]
        full_filename = file
        list_of_regex_matches.append(RegexMatch(match_idx_str = idx, full_filename = full_filename))
    return list_of_regex_matches


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "Convert a knowledge graph dataset to training JSONs containing image filenames, metadata filenames, and ground_truth_y attributes")

    parser.add_argument("--folder_to_parse", default = "/knowledge_graph/example_single_node_dataset")
    parser.add_argument("--output_json_full_path", default = "/knowledge_graph/training_data_jsons/data.json")

    args_cli = parser.parse_args()

    convert_dataset_to_evaluation_datapoints(folder = args_cli.folder_to_parse,
                                             output_json_full_path = args_cli.output_json_full_path)
