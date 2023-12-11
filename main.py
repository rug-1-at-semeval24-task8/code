import argparse
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed

from bilstm import BiLSTM
from features.perplexity import PerplexityFeature
from features.predictability import PredictabilityFeature
from training import eval_loop, train_loop


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

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    local_device = torch.device("cpu")

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
    torch.manual_seed(10)
    random.seed(10)
    torch.manual_seed(10)
    np.random.seed(0)
    
    out_path = pathlib.Path("/home/huseyin/repo/code") / "out"

    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    train_df = train_df.head(128)
    valid_df = valid_df.head(128)
    test_df = test_df.head(128)

    train_documents = train_df["text"].tolist()
    valid_documents = valid_df["text"].tolist()
    test_documents = test_df["text"].tolist()

    train_ids = [str(x) for x in train_df["id"].tolist()]
    dev_ids = [str(x) for x in valid_df["id"].tolist()]
    test_ids = [str(x) for x in test_df["id"].tolist()]

    # pp_feature = PerplexityFeature(device=device, local_device=local_device, model_id="gpt2")
    pred_feature = PredictabilityFeature(
        device=device, local_device=local_device, language="en"
    )
    # grammar_feature = GrammarFeatures(device=device, local_device=local_device, language="en")
    # word_freq_feature = WordFrequency(device=device, local_device=local_device, language="en")

    #  Add your features instances

    featurizers = [pred_feature]  # extend list with your features

    train_X = []
    dev_X = []
    test_X = []
    for fz in featurizers:
        train_X.append(np.array(fz.features(train_documents)))
        dev_X.append(np.array(fz.features(valid_documents)))
        test_X.append(np.array(fz.features(test_documents)))

    train_X = np.concatenate(train_X, axis=2)
    dev_X = np.concatenate(dev_X, axis=2)
    test_X = np.concatenate(test_X, axis=2)

    if subtask == "A":
        train_Y = np.array(train_df["label"].tolist())
        dev_Y = np.array(valid_df["label"].tolist())
    elif subtask == "B":
        train_Y = np.array([label2id[x] for x in train_df["label"].tolist()])
        dev_Y = np.array([label2id[x] for x in valid_df["label"].tolist()])

    print("Building a model...")
    BATCH_SIZE = 16
    train_dataset = TensorDataset(
        torch.tensor(train_X).float(),
        torch.tensor(np.array(train_Y)).long(),
    )
    dev_dataset = TensorDataset(
        torch.tensor(dev_X).float(),
        torch.tensor(np.array(dev_Y)).long(),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_X).float(),
        torch.tensor(np.zeros(len(test_documents))).long(),
    )
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

    model = BiLSTM(train_X.shape[2], "A", local_device).to(device)

    language = "en"

    
    stats_path = out_path / (subtask + "_" + language + "_stats.tsv")
    model_type = "bilstm"
    
    print("Preparing training")
    model = model.to(device)
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    milestones = [5] if model_type == "Hybrid" else []
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.02)
    skip_visual = False
    stats_file = open(stats_path, "w")
    stats_file.write("epoch\ttrain_F1\tdev_F1\n")

    eval_loop(dev_loader, model, device, local_device, skip_visual)
    for epoch in range(20):
        print("EPOCH " + str(epoch + 1))

        train_f1 = train_loop(
            train_loader, model, optimizer, scheduler, device, local_device, skip_visual
        )
        dev_f1 = eval_loop(
            dev_loader, model, device, local_device, skip_visual, test=False
        )
        test_preds, test_probs = eval_loop(
            test_loader, model, device, local_device, skip_visual, test=True
        )

        stats_file.write(
            str(epoch + 1) + "\t" + str(train_f1) + "\t" + str(dev_f1) + "\n"
        )

        with open(
            out_path / (subtask + "_" + language + "_preds_" + str(epoch + 1) + ".tsv"),
            "w",
        ) as f:
            f.write("id\tlabel\n")
            for test_id, pred in zip(test_ids, test_preds):
                label = (
                    ["human", "generated"][pred]
                    if subtask == "A"
                    else ["A", "B", "C", "D", "E", "F"][pred]
                )
                f.write(test_id + "\t" + label + "\n")

        with open(
            out_path
            / (subtask + "_" + language + "_probs_test_" + str(epoch + 1) + ".tsv"),
            "w",
        ) as f:
            f.write(
                "id\t"
                + "\t".join(
                    ["human", "generated"]
                    if subtask == "A"
                    else ["A", "B", "C", "D", "E", "F"]
                )
                + "\n"
            )
            for test_id, prob in zip(test_ids, test_probs):
                f.write(test_id + "\t" + "\t".join([str(x) for x in prob]) + "\n")

        _, dev_probs = eval_loop(
            dev_loader, model, device, local_device, skip_visual, test=True
        )
        _, train_probs = eval_loop(
            train_loader, model, device, local_device, skip_visual, test=True
        )
        with open(
            out_path
            / (subtask + "_" + language + "_probs_traindev_" + str(epoch + 1) + ".tsv"),
            "w",
        ) as f:
            f.write(
                "id\t"
                + "\t".join(
                    ["human", "generated"]
                    if subtask == "A"
                    else ["A", "B", "C", "D", "E", "F"]
                )
                + "\n"
            )
            for test_id, prob in zip(
                train_ids + dev_ids, [x for x in train_probs] + [x for x in dev_probs]
            ):
                f.write(test_id + "\t" + "\t".join([str(x) for x in prob]) + "\n")

    stats_file.close()