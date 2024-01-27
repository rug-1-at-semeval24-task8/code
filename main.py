import comet_ml  # NOTE : Always import comet_ml in the top of your file
import argparse
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd
import torch
from comet_ml import Experiment
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import set_seed

from bilstm import BiLSTM
from dataset import TensorTextDataset
from features.perplexity import PerplexityFeature
from features.predictability import PredictabilityFeature
from training import eval_loop, train_loop

experiment = Experiment(
    api_key="nLqFerDLnwvCiAptbL4u0FZIj",
    project_name="shared-task",
    workspace="halecakir",
)


def get_data(
    train_path,
    test_path,
    random_seed
):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["label"],
        random_state=random_seed,
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
    parser.add_argument("--mono", "-mono", help="Monolingual", action="store_true")
    parser.add_argument(
        "--models",
        "-m",
        required=False,
        help="Transformer to train and test",
        default=["xlm-roberta-base"],
        nargs="+",
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

    parser.add_argument("--batch_size", "-bs", help="Batch size", default=16)
    parser.add_argument(
        "--batch_size_feature_extraction",
        "-bsfe",
        help="Batch size",
        default=32,
        type=int,
    )
    parser.add_argument("--epochs", "-e", help="Epochs", default=20, type=int)
    parser.add_argument(
        "--fixed_length", "-fl", help="Fixed length", default=128, type=int
    )
    parser.add_argument("--seed", "-s", help="Seed", default=10, type=int)
    parser.add_argument(
        "--enable_preditability",
        "-ep",
        help="Enable predictability feature",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--enable_perplexity",
        "-epp",
        help="Enable perplexity feature",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--enable_information_redundancy",
        "-eir",
        help="Enable information redundancy features",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--enable_entity_coherence",
        "-eec",
        help="Enable entity coherence features (note: this should only be enabled for monolingual English data)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--data_size", "-ds", help="Data size", default=-1, type=int)
    parser.add_argument("--tags", "-tg", help="Tags", nargs="+", default=[])
    parser.add_argument(
        "--hidden_size", "-hs", default=64, type=int, help="Hidden size"
    )
    parser.add_argument(
        "--num_lstm_layers", "-nl", default=2, type=int, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout_prop", "-dp", default=0.0, type=float, help="Dropout proportion"
    )
    parser.add_argument(
        "--attention_enabled",
        "-ae",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Attention enabled",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    local_device = torch.device("cpu")

    random_seed = args.seed
    train_path = args.train_file_path  # For example 'subtaskA_train_multilingual.jsonl'
    test_path = args.test_file_path  # For example 'subtaskA_test_multilingual.jsonl'
    models = args.models  # For example 'xlm-roberta-base'
    subtask = args.subtask  # For example 'A'
    prediction_path = (
        args.prediction_file_path
    )  # For example subtaskB_predictions.jsonl
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_lstm_layers = args.num_lstm_layers
    dropout_prop = args.dropout_prop
    attention_enabled = args.attention_enabled

    # LOG PARAMETERS
    experiment.log_parameters(vars(args))
    experiment.add_tags(args.tags)

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
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    if subtask == "A":
        experiment.add_tag("subtask_a")
        if args.mono:
            experiment.add_tag("monolingual")
            out_path = "subtask_a_monolingual.jsonl"
        else:
            experiment.add_tag("monolingual")
            out_path = "subtask_a_multilingual.jsonl"
    elif subtask == "B":
        experiment.add_tag("subtask_b")
        out_path = "subtask_b.json"
    elif subtask == "C":
        experiment.add_tag("subtask_c")
        out_path = "subtask_c.json"

    train_df, valid_df, test_df = get_data(
        train_path,
        test_path,
        random_seed
    )

    # for testing purposes
    if args.data_size > 0:
        train_df = train_df.head(args.data_size)
        valid_df = valid_df.head(args.data_size)
    


    train_documents = train_df["text"].tolist()
    valid_documents = valid_df["text"].tolist()
    test_documents = test_df["text"].tolist()

    train_ids = [str(x) for x in train_df["id"].tolist()]
    dev_ids = [str(x) for x in valid_df["id"].tolist()]
    test_ids = [str(x) for x in test_df["id"].tolist()]

    #  Add your features instances
    featurizers = []
    doc_level_featurizers = []
    if args.enable_preditability:
        pred_feature = PredictabilityFeature(
            device=device,
            local_device=local_device,
            batch_size=args.batch_size_feature_extraction,
            fixed_length=args.fixed_length,
            experiment=experiment,
            models=models,  # , "distilgpt2", "gpt2-large", "gpt2-xl"],
        )
        featurizers.append(pred_feature)
    if args.enable_perplexity:  # Example doc level feature
        pp_feature = PerplexityFeature(
            device=device,
            local_device=local_device,
            fixed_length=args.fixed_length,
            experiment=experiment,
            models=models,  # , "distilgpt2", "gpt2-large", "gpt2-xl"],
        )
        doc_level_featurizers.append(pp_feature)
    if args.enable_information_redundancy:
        from features.information_redundancy import InformationRedundancyFeature

        info_red_feature = InformationRedundancyFeature(
            device=device, local_device=local_device
        )
        doc_level_featurizers.append(info_red_feature)
    if args.enable_entity_coherence:
        from features.entity_coherence import EntityCoherenceFeature

        entity_coh_feature = EntityCoherenceFeature(
            device=device, local_device=local_device
        )
        doc_level_featurizers.append(entity_coh_feature)
    # NOTE: Add your doc level features here as we did for perplexity feature

    train_X = []
    dev_X = []
    test_X = []
    for fz in featurizers:
        train_X.append(np.array(fz.features(train_documents)))
        dev_X.append(np.array(fz.features(valid_documents)))
        test_X.append(np.array(fz.features(test_documents)))

    doc_train_X = []
    doc_dev_X = []
    doc_test_X = []
    for fz in doc_level_featurizers:
        doc_train_X.append(np.array(fz.features(train_documents)))
        doc_dev_X.append(np.array(fz.features(valid_documents)))
        doc_test_X.append(np.array(fz.features(test_documents)))

    train_X = np.concatenate(train_X, axis=2)
    dev_X = np.concatenate(dev_X, axis=2)
    test_X = np.concatenate(test_X, axis=2)

    doc_train_X = np.concatenate(doc_train_X, axis=1)
    doc_dev_X = np.concatenate(doc_dev_X, axis=1)
    doc_test_X = np.concatenate(doc_test_X, axis=1)

    if subtask == "A":
        train_Y = np.array(train_df["label"].tolist())
        dev_Y = np.array(valid_df["label"].tolist())
        # Remove it later:
        
    elif subtask == "B":
        train_Y = np.array(train_df["label"].tolist())
        dev_Y = np.array(valid_df["label"].tolist())
        # Remove it later:
        

    print("Building a model...")

    train_dataset = TensorTextDataset(
        torch.tensor(train_X).float(),
        torch.tensor(np.array(train_Y)).long(),
        doc_train_X,
    )
    dev_dataset = TensorTextDataset(
        torch.tensor(dev_X).float(), torch.tensor(np.array(dev_Y)).long(), doc_dev_X
    )
    test_dataset = TensorTextDataset(
        torch.tensor(test_X).float(), torch.tensor(np.array(list(range(test_X.shape[0])))).long(), doc_test_X
    )
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    model = BiLSTM(
        train_X.shape[2],
        doc_train_X.shape[1],
        subtask,
        local_device,
        device,
        hidden_size,
        num_lstm_layers,
        dropout_prop,
        attention_enabled,
    )

    print("Preparing training")

    model = model.to(device)
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    milestones = [5] 
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.02)
    skip_visual = False

    eval_loop(dev_loader, model, device, local_device, skip_visual)
    for epoch in range(args.epochs):
        experiment.set_epoch(epoch + 1)
        print("EPOCH " + str(epoch + 1))

        train_loss, train_f1 = train_loop(
            train_loader, model, optimizer, scheduler, device, local_device, skip_visual
        )

        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("train_f1", train_f1)

        print("Development set evaluation")
        dev_f1_micro, dev_f1_macro, dev_loss = eval_loop(
            dev_loader, model, device, local_device, skip_visual
        )

        experiment.log_metric("dev_f1_macro", dev_f1_macro)
        experiment.log_metric("dev_f1_micro", dev_f1_micro)
        experiment.log_metric("dev_loss", dev_loss)

    print("Test set evaluation")        
    preds, probs = eval_loop(test_loader, model, device, local_device, skip_visual, test=True)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': preds})
    predictions_df.to_json(out_path, lines=True, orient='records')