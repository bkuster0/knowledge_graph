# Knowledge Graph dataset


# Overview
Dataset of images & Question-Answer pairs in JSON format, as well as segmentation data.


This dataset can by utilized for [Retrieval-Augmented Generation](https://repo.ijs.si/bkuster/rag_retrieval_augmented_generation).

# Graph dataset format:

    training_data_folder/ 
    ├── graph_edges.csv # File containing the connections between nodes in the graph. 
    └── nodes/ # Directory containing data for each node. 
        └── device_0/ # Directory for a specific node, e.g., device_0. 
            ├── 0.jpg # Image associated with this node. 
            ├── 0.json # Segmentation data in LabelMe JSON format. 
            ├── 0_qa.json # Question-Answer pairs in JSON.
            └── 1.jpg # Second image associated with this node.

**graph\_edges.csv** should contain comma-separated **parent\_node,child\_node** connections, e.g.
> device,smoke\_detector

# Single folder (node definition)

A folder (node) should contain a set of images (e.g. **0.jpg**). Each image should have a corresponding Question-Answer JSON (**0_qa.json**) and a metadata json (**0_meta.json**).


Example **QA JSON**:


    [
        {
            "Q": "What device is shown?",
            "A": "Smoke detector Hekatron."
        },
        {
            "Q": "What is the next disassembly step?",
            "A": "Place into the CNC mill."
        }
    ]


Example **metadata JSON**:

    {
        "number_of_remaining_steps": 3,
        "is_final_step": 0
    }



# Conversion to dataset

Convert to dataset for training a classifier:

>     cd scripts
>     python3 convert_dataset_to_training_data.py --folder_to_parse '/knowledge_graph/datasets/train_knowledge_graph' --output_classification_json_full_path '/knowledge_graph/training_data_jsons/classification_data.json' --output_vlm_json_full_path '/knowledge_graph/training_data_jsons/vlm_data.json'

For the example dataset:

>     cd scripts
>     python3 convert_dataset_to_training_data.py --folder_to_parse '/knowledge_graph/example_single_node_dataset' --output_classification_json_full_path '/knowledge_graph/training_data_jsons/classification_data.json' --output_vlm_json_full_path '/knowledge_graph/training_data_jsons/vlm_data.json'

# OLD - Misc

Device components in image form can be found in: `~/datasets2/reconcycle/2023-05-23_synthetic_dataset/foregrounds_saved`. They are generated by [get_foregrounds.ipynb](https://gitlab.gwdg.de/sebastian.ruiz/synthetic-dataset-creator/-/blob/master/get_foregrounds.ipynb).

New images can be labelled using [Labelme with segment anything](https://github.com/originlake/labelme-with-segment-anything).

0\_qa.json files can be created using [knowledge_graph_generator.ipynb](https://github.com/ReconCycle/vision_pipeline/blob/dev/notebooks/knowledge_graph_generator.ipynb).

