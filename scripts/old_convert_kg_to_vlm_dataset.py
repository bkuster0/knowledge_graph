""" Dataset Format:
https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json

"mllm_demo": {
    "file_name": "mllm_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
"""


import copy
import os
import json
from typing import List, Union
from rag_retrieval_augmented_generation.read_knowledge_graph_dataset import parse_kg_folder


def convert_to_string(answer):
    if isinstance(answer, dict):
        return str(answer)
        #return json.dumps(json.loads(answer))
    return answer


def single_knowledge_graph_node_to_sharegpt_data(parsed_kg_folder):
    """
    >>> from rag_retrieval_augmented_generation.file_utils import parse_kg_folder
    >>> parsed_kg_folder = parse_kg_folder('/knowledge_graph/data/nodes/', 'kalo')
    >>> single_knowledge_graph_node_to_sharegpt_data(parsed_kg_folder)
    """
    output = []
    single_conv_template = {"messages": []}
    image_token = '<image>'
    # Conversion between KG's QA format to sharegpt
    mapping = {"Q": {"role": "user", "content": "placeholder"},
               "A": {"role": "assistant", "content": "placeholder"}}

    text_fns_to_img_fns = parsed_kg_folder.connections

    node_name = parsed_kg_folder.node_name
    full_folder_path_with_node_name = os.path.join(parsed_kg_folder.folder_path, node_name)

    # If no text QAs are present, ain't much we can do. return empty list
    if len(parsed_kg_folder.texts.keys()) == 0:
        #print(f"Node folder '{full_folder_path_with_node_name}' does not contain any useful data")
        return output

    for text_fn, text_qa in parsed_kg_folder.texts.items():
        print("TEXT FNS TO IMG FNS", text_fns_to_img_fns)
        image_included = False
        tmp_image_token = '' # By default empty, no image
        if text_fn in text_fns_to_img_fns.keys():
            image_included = True
            img_fn = text_fns_to_img_fns[text_fn]
            img_fn = os.path.join(full_folder_path_with_node_name, img_fn)
            tmp_image_token = image_token
        for q_a_dict in text_qa:
            q = q_a_dict["Q"]
            a = q_a_dict["A"]
            q_msg = copy.deepcopy(mapping["Q"])
            q_msg["content"] = tmp_image_token + q
            a_msg = copy.deepcopy(mapping["A"])

            # Check if answer is JSON and convert into string
            a_msg["content"] = convert_to_string(a)
            #a_msg["content"] = json.dumps(a)
            #a_msg["content"] = a
            single_conv = copy.deepcopy(single_conv_template)
            single_conv["messages"].append(q_msg)
            single_conv["messages"].append(a_msg)
            
        if image_included:
            single_conv["images"] = [img_fn]

        output.append(single_conv)
            
    0
    return output

def knowledge_graph_to_sharegpt_dataset(knowledge_graph_folder = '/knowledge_graph/',
                                       output_json_full_path = '/catkin_ws/src/allnet/allnet/saved_models_logs_outputs/datasets/dataset.json'):
    """
    >>> out = knowledge_graph_to_sharegpt_dataset(knowledge_graph_folder = '/knowledge_graph/',
                                                  output_json_full_path = '/catkin_ws/src/allnet/allnet/saved_models_logs_outputs/datasets/dataset.json')
    """
    out_list = []

    folders = os.walk(knowledge_graph_folder)
    for (root,dirs,files) in os.walk(knowledge_graph_folder):
        #print(root)
        # Ignore hidden folders
        if '.' in root:
            #print("Hidden folder:", root)
            continue
        # Ignore the outer-most folder with no data elements
        split = os.path.split(root)
        if len(split[1]) == 0:
            #print("ignoring", split)
            continue

        # Try to parse folder
        base_folder = split[0]
        node_folder = split[1]
        #print(base_folder, node_folder)
        try:
            o = parse_kg_folder(folder_path = root)
            #print(o)
        except Exception as e:
            print(e)
        #print(o)

        sharegpt_node_data = single_knowledge_graph_node_to_sharegpt_data(o)
        out_list.extend(sharegpt_node_data)

    if output_json_full_path is not None:
        with open(output_json_full_path, 'w') as f:
            json.dump(out_list, f, indent = 2)

    return out_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "Convert a knowledge graph dataset to training JSONs containing image filenames, metadata filenames, and ground_truth_y attributes")

    parser.add_argument("--folder_to_parse", default = "/knowledge_graph/example_single_node")
    parser.add_argument("--output_json_full_path", default = "/knowledge_graph/training_data_jsons/vlm_data.json")

    args_cli = parser.parse_args()

    knowledge_graph_to_sharegpt_dataset(knowledge_graph_folder = args_cli.folder_to_parse,
                                                  output_json_full_path = args_cli.output_json_full_path)

