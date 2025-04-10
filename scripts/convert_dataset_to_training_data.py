import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Union
import json
import copy
import random


@dataclass
class EvaluationDatapoint:
    step_idx: int = 0 # The index of step (0 - first step e.g. "flip", 1- second step e.g. "move_to_cnc")
    step_example_idx: int = 0 # Img example of single step (in case of more images, e.g. step 0, img 0; step 0, img 1)
    full_image_path: str = '/knowledge_graph/example_single_node_dataset/smoke_detector/0_0.jpg'
    full_metadata_path: str = '/knowledge_graph/example_single_node_dataset/smoke_detector/seq_metadata.json'

    object_general_class: str = "heat_cost_allocator"
    object_specific_subclass: str = "kalo"
    previous_step: str = "move"
    next_step: str = "flip"
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

# Full dataset -> evaluation Datapoints
#(convert_sequence_metadata_to_evaluation_datapoints)

# Evaluation datapoints -> QA Datapoints
#(convert_evaluation_datapoints_to_qa_datapoints)

# QA Datapoints -> 
    # - VLM training data
        # (convert_qa_datapoints_to_vlm_training_json)
    # - RAG QA JSONs
        # (convert_qa_datapoints_to_rag_json)


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

        if is_folder_hidden(root):
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
                                    list_of_strings: List[str] = ['0_0.jpg']) -> List[RegexMatch]:
    list_of_regex_matches = []

    for file in list_of_strings:
        match = extension_regex_pattern.search(file)
        if match is None:
            continue
        idx = file[:match.start()]
        full_filename = file
        #print(f"regex: idx: {idx}, file: {file}, match_start: {match.start()}, match_end: {match.end()}")
        list_of_regex_matches.append(RegexMatch(match_idx_str = idx, full_filename = full_filename))
    return list_of_regex_matches


def convert_sequence_metadata_to_evaluation_datapoints(folder, output_json_full_path = None) -> List[EvaluationDatapoint]:
    """
    Read a dataset or single node (walks all folders within folder) to output evaluation datapoints.
    (Finds matches between images and metadata and packs them into EvaluationDatapoint class).

    Example use:
    >>> eval_datapoints = convert_sequence_metadata_to_evaluation_datapoints(folder = '/knowledge_graph/example_single_node_dataset/', output_json_full_path = '/knowledge_graph/training_data_jsons/data.json')

    Args:
    folder: Folder within which all folders will be walked (os.walk)

    Returns: List[EvaluationDatapoint]

   
    """

    IMAGE_EXTENSION_REGEX = '\.(png|jpg|jpeg)'
    metadata_fn = 'seq_metadata.json'

    image_extension_re = re.compile(IMAGE_EXTENSION_REGEX)

    evaluation_datapoints = []
    for (root,dirs,files) in os.walk(folder,topdown=True):
        if is_folder_hidden(root):
            continue

        # Find all image files
        image_regex_matches = filter_files_by_regex_extension(extension_regex_pattern = image_extension_re,
                                                              list_of_strings = files)

        # Find metadata file. Continue if not exists
        full_metadata_path = os.path.join(root, metadata_fn)
        if not os.path.exists(full_metadata_path):
            #print(f"seq_metadata.json not found in folder {folder}")
            continue

        # Load metadata
        with open(full_metadata_path, 'r') as f:
            try:
                metadata_dict = json.load(f)
            except Exception as e:
                raise ValueError(f"Exception {e}. File: {full_metadata_path}")
            #print(metadata_dict)

        #print("Folder", root)
        #print("Images", image_regex_matches)
        #print(files)
        #print("\n\n")
        for regex_match in image_regex_matches:
            idx_str = regex_match.match_idx_str
            full_filename = regex_match.full_filename

            step_n = int(idx_str.split('_')[0]) # index of step - step 0, step 1
            example_img_idx = int(idx_str.split('_')[1]) # index of image showing step e.g. 0 - image 0, image 1, ...

            full_image_path = os.path.join(root, full_filename)

            object_general_class = metadata_dict["object_general_class"]
            object_specific_subclass = metadata_dict["object_specific_subclass"]
            next_step = metadata_dict["steps"][step_n]
            n_remaining_steps = len(metadata_dict["steps"]) - step_n
            is_final_step = True if step_n == len(metadata_dict["steps"]) - 1 else False
            previous_step =  metadata_dict["steps"][step_n - 1] if not is_final_step else "None"
            print(f"Image: {full_filename}\nn_remaining_steps: {n_remaining_steps}\nobject_general_class: {object_general_class}\nNext step: {next_step}, is_final_step: {is_final_step}")
            print(f"Previous step: {previous_step}")

            evaluation_datapoint = EvaluationDatapoint(step_idx = step_n,
                                                       step_example_idx = example_img_idx,
                                                       full_image_path = full_image_path,
                                                       full_metadata_path = full_metadata_path,
                                                       object_general_class = object_general_class,
                                                       object_specific_subclass = object_specific_subclass,
                                                       previous_step = previous_step,
                                                       next_step = next_step,
                                                       number_of_remaining_steps = n_remaining_steps,
                                                       is_final_step = is_final_step)

            evaluation_datapoints.append(evaluation_datapoint)

    # Save as JSON
    if output_json_full_path is not None:
        datapoints_dict = [datapoint.to_dict() for datapoint in evaluation_datapoints]
        json_data = json.dumps(datapoints_dict, indent = 4)
        with open(output_json_full_path, 'w') as json_file:
            json_file.write(json_data)

    return evaluation_datapoints

def convert_evaluation_datapoints_to_qa_datapoints(evaluation_datapoints = List[EvaluationDatapoint],
                                                   json_definitions_path = "questions_and_answers_to_vlm_randomized_qa.json") -> List[QADatapoint]:
    with open(json_definitions_path, 'r') as f:
        qa_to_randomized_vlm_qa = json.load(f)

    out_qa_datapoints = []

    for eval_datapoint in evaluation_datapoints:
        metadata_path = eval_datapoint.full_metadata_path
        image_path = eval_datapoint.full_image_path

        out_qa_pairs = []
        # q - question, a - answer
        # Try to convert each variable in EvaluationDatapoint to a question
        for q in vars(eval_datapoint).keys():
            #print("q:", q)
            # Randomize question
            if q not in qa_to_randomized_vlm_qa.keys():
                #raise ValueError(f"Question '{q}' not specified in qa_to_randomized_vlm_qa dict!")
                continue
            n_random_possibilities = len(qa_to_randomized_vlm_qa[q]["questions"])
            random_idx = random.randint(0, n_random_possibilities -1 )
            randomized_question = qa_to_randomized_vlm_qa[q]["questions"][random_idx]
            #print(randomized_question)

            # Randomize answer
            answer = str(getattr(eval_datapoint, q))
            print(q, answer)
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
            else:
                print(f"Did not find answer mapping for q: {q}, answer: {answer}")

            # Convert answer to string if it's integer
            #if isinstance(answer, int):
            #    answer = str(answer)

            out_qa_pairs.append({"Q": randomized_question, "A": answer})

        qa_datapoint = QADatapoint(full_image_path = image_path,
                                   list_of_qa_pairs = out_qa_pairs)

        out_qa_datapoints.append(copy.deepcopy(qa_datapoint))
    return out_qa_datapoints


def is_folder_hidden(folder = "/knowledge_graph/.test_hidden"):
    """
    Check if a particular path/folder is hidden (starts with . in linux). 
    Returns true if any part of the path is hidden, not just outermost folder.

    >>> hidden = is_folder_hidden('/.test')
    >>> hidden = is_folder_hidden('/test')
    """
    splits = os.path.split(folder)
    is_split_hidden = [split.startswith('.') for split in splits]
    folder_is_hidden = True if True in is_split_hidden else False
    return folder_is_hidden

def convert_qa_datapoints_to_rag_json(qa_datapoints: List[QADatapoint] = None,
                                      save_json = True):

    json_file_suffix = "_qa.json"
    all_jsons = []

    for datapoint in qa_datapoints:
        qa_pairs = datapoint.list_of_qa_pairs
        out_json = []
        # Add questions and answers
        for qa_pair in qa_pairs:
            q = qa_pair["Q"]
            a = qa_pair["A"]

            dic = {"Q": q, "A": a}
            out_json.append(dic)

        # Save json
        if save_json:
             #Determine save path
            #print(datapoint.full_image_path)
            img_filename = os.path.split(datapoint.full_image_path)[-1]
            file_idx = img_filename.split('.')[0]
            json_filename = file_idx + json_file_suffix
            folder = os.path.dirname(datapoint.full_image_path)
            full_json_path = os.path.join(folder, json_filename)
            #print(full_json_path)
            with open(full_json_path, 'w') as f:
                json.dump(out_json, f, indent = 4)
                print(f"Saving JSON to : {full_json_path}")
        all_jsons.append(out_json)
    return all_jsons


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "Convert a knowledge graph dataset to training JSONs containing image filenames, metadata filenames, and ground_truth_y attributes")

    parser.add_argument("--folder_to_parse", default = "/knowledge_graph/example_single_node_dataset")
    parser.add_argument("--output_classification_json_full_path", default = "/knowledge_graph/training_data_jsons/classification_data.json")
    parser.add_argument("--output_vlm_json_full_path", default = "/knowledge_graph/training_data_jsons/vlm_data.json")

    args_cli = parser.parse_args()

    eval_datapoints = convert_sequence_metadata_to_evaluation_datapoints(folder = args_cli.folder_to_parse, output_json_full_path = args_cli.output_classification_json_full_path)
    qa_datapoints = convert_evaluation_datapoints_to_qa_datapoints(evaluation_datapoints = eval_datapoints)

    convert_qa_datapoints_to_vlm_training_json(qa_datapoints = qa_datapoints, output_json_full_path = args_cli.output_vlm_json_full_path)

    out = convert_qa_datapoints_to_rag_json(qa_datapoints = qa_datapoints, save_json = True)
