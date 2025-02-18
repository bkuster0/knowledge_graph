import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Union
import json
import copy
import random


@dataclass
class EvaluationDatapoint:
    full_image_path: str = '/knowledge_graph/example_single_node_dataset/smoke_detector/0.jpg'
    full_metadata_path: str = '/knowledge_graph/example_single_node_dataset/smoke_detector/0_meta.json'
    number_of_remaining_steps: int = 1
    is_final_step: int = 0

    def to_dict(self):
        return vars(self)


@dataclass
class QADatapoint:
    full_image_path: str = '/0.jpg'
    list_of_qa_pairs: List[Dict[str, str]] = field(
            default_factory=lambda: [{"Q": "What is the number of remaining disassembly steps?",
                                      "A": "3"}
                                    ]
        )


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


def convert_evaluation_datapoints_to_vlm_qa_datapoints(evaluation_datapoints = List[EvaluationDatapoint]) -> List[QADatapoint]:
    JSON_FILE_PATH = "questions_and_answers_to_vlm_randomized_qa.json"
    with open(JSON_FILE_PATH, 'r') as f:
        qa_to_randomized_vlm_qa = json.load(f)

    out_qa_datapoints = []

    for eval_datapoint in evaluation_datapoints:
        metadata_path = eval_datapoint.full_metadata_path
        image_path = eval_datapoint.full_image_path

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        out_qa_pairs = []
        # q - question, a - answer
        for q in metadata.keys():
            # Randomize question
            if q not in qa_to_randomized_vlm_qa.keys():
                raise ValueError(f"Question '{q}' not specified in qa_to_randomized_vlm_qa dict!")
            n_random_possibilities = len(qa_to_randomized_vlm_qa[q]["questions"])
            random_idx = random.randint(0, n_random_possibilities -1 )
            randomized_question = qa_to_randomized_vlm_qa[q]["questions"][random_idx]
            #print(randomized_question)

            # Randomize answer
            answer = str(metadata[q])
            #if not (isinstance(answer, str) or isinstance(answer, int)):
            #    raise ValueError(f"Unsupported answer type, neither string not integer. Answer is: {answer}")

            # Try to convert answer to one of randomized answers.
            if "answers" in qa_to_randomized_vlm_qa[q].keys() and (answer in qa_to_randomized_vlm_qa[q]["answers"].keys()):
                answer_to_randomized_answer = qa_to_randomized_vlm_qa[q]["answers"][answer]
                # If only single choice, not a list of choices
                if isinstance(answer_to_randomized_answer, str):
                    answer = answer_to_randomized_answer
                else:
                    #print("ans to possible ans:", answer_to_randomized_answer)
                    n_possible_answers = len(answer_to_randomized_answer)
                    randomized_answer = answer_to_randomized_answer[random.randint(0, n_possible_answers -1)]
                    answer = randomized_answer
            # Convert answer to string if it's integer
            #if isinstance(answer, int):
            #    answer = str(answer)

            out_qa_pairs.append({"Q": randomized_question, "A": answer})

        qa_datapoint = QADatapoint(full_image_path = image_path,
                                   list_of_qa_pairs = out_qa_pairs)

        out_qa_datapoints.append(copy.deepcopy(qa_datapoint))
    return out_qa_datapoints


def convert_qa_datapoints_to_vlm_training_json(qa_datapoints: List[QADatapoint],
                                               output_json_full_path = None):
    ADDITIONAL_IMAGE_TOKEN = "<image>"
    out_json = []

    for datapoint in qa_datapoints:
        img_path = datapoint.full_image_path
        qa_pairs = datapoint.list_of_qa_pairs
        tmp_image_token = ADDITIONAL_IMAGE_TOKEN if img_path is not None else ''
        tmp_dict = {"messages": []}

        # Add questions and answers
        for qa_pair in qa_pairs:
            q = qa_pair["Q"]
            a = qa_pair["A"]
            tmp_dict["messages"].append(
            {
                "role": "user",
                "content": f"{tmp_image_token}{q}"
            })
            tmp_dict["messages"].append(
            {
                "role": "assistant",
                "content": f"{a}"
            }
            )
        # Add image
        if img_path is not None:
            tmp_dict["images"] = [img_path]

        out_json.append(tmp_dict)

    if output_json_full_path is not None:
        json_data = json.dumps(out_json, indent = 4)
        with open(output_json_full_path, 'w') as f:
            f.write(json_data)
    return out_json

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
    parser.add_argument("--output_classification_json_full_path", default = "/knowledge_graph/training_data_jsons/classification_data.json")
    parser.add_argument("--output_vlm_json_full_path", default = "/knowledge_graph/training_data_jsons/vlm_data.json")

    args_cli = parser.parse_args()

    evaluation_datapoints = convert_dataset_to_evaluation_datapoints(folder = args_cli.folder_to_parse, output_json_full_path = args_cli.output_classification_json_full_path)
    qa_datapoints = convert_evaluation_datapoints_to_vlm_qa_datapoints(evaluation_datapoints = evaluation_datapoints)
    convert_qa_datapoints_to_vlm_training_json(qa_datapoints = qa_datapoints, output_json_full_path = args_cli.output_vlm_json_full_path)
