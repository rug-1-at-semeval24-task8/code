import argparse
import pandas as pd
from transformers import set_seed
import os
from sklearn.model_selection import train_test_split
import argparse
import logging
from features.perplexity import PerplexityFeature
import numpy as np


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=random_seed
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file_path",
        "-tr",
        required=True,
        help="Path to the train file.",
        type=str,
    )
    parser.add_argument(
        "--test_file_path", "-t", required=True, help="Path to the test file.", type=str
    )
    parser.add_argument(
        "--subtask",
        "-sb",
        required=True,
        help="Subtask (A or B).",
        type=str,
        choices=["A", "B"],
    )
    parser.add_argument(
        "--model", "-m", required=True, help="Transformer to train and test", type=str
    )
    parser.add_argument(
        "--prediction_file_path",
        "-p",
        required=True,
        help="Path where to save the prediction file.",
        type=str,
    )
    parser.add_argument(
        "--feat_perplexity", "-fperp", help="Feature perplexity", default=True
    )

    args = parser.parse_args()

    random_seed = 0
    train_path = args.train_file_path  # For example 'subtaskA_train_multilingual.jsonl'
    test_path = args.test_file_path  # For example 'subtaskA_test_multilingual.jsonl'
    model = args.model  # For example 'xlm-roberta-base'
    subtask = args.subtask  # For example 'A'
    prediction_path = (
        args.prediction_file_path
    )  # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    if subtask == "A":
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == "B":
        id2label = {
            0: "human",
            1: "chatGPT",
            2: "cohere",
            3: "davinci",
            4: "bloomz",
            5: "dolly",
        }
        label2id = {
            "human": 0,
            "chatGPT": 1,
            "cohere": 2,
            "davinci": 3,
            "bloomz": 4,
            "dolly": 5,
        }
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    train_documents = train_df["text"].tolist()
    valid_documents = valid_df["text"].tolist()
    test_documents = test_df["text"].tolist()

    pp_feature = PerplexityFeature(device="cpu", model_id="gpt2")
    #  Add your features instances

    featurizers = [pp_feature] # extend list with your features

    train_X = []
    for fz in featurizers:
        train_X.append(np.array(fz.features(train_documents[:100])))

    train_X = np.concatenate(train_X, axis=2) # concatenated features
    

    # python main.py --train_file_path SubtaskA/subtaskA_train_monolingual.jsonl --test_file_path SubtaskA/subtaskA_dev_monolingual.jsonl --prediction_file_path result --subtask A --model A
