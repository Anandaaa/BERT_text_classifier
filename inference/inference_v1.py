#!/usr/bin/env python3


import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from json import load

import numpy as np
from torch.nn import Sigmoid
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification
from transformers import AutoTokenizer


def main(raw_args: List[str]):
    args = parse_raw_arguments(raw_args)
    with args.model_dir.joinpath("config.json").open("r") as file:
        model_config = load(file)
    #with args.model_dir.joinpath("id2label.json").open("r") as file:
    #    id2label = load(file)
    #    id2label = {int(k): v for k, v in id2label.items()}
    id2label = {int(k): v for k, v in model_config["id2label"].items()}
    print(f"id2label: {id2label}")
    label2id = {v: k for k, v in id2label.items()}
    tokenizer_path = args.model_dir.joinpath("tokenizer.json")
    logging.info(f"Loading tokanizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    #tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    logging.info(f"Loading model from: {args.model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        problem_type="multi_label_classification",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    encoding = tokenizer(args.text, return_tensors="pt")
    encoding = {k: v.to(model.device) for k, v in encoding.items() if k in ["input_ids", "attention_mask"]}
    print(encoding)
    outputs = model(**encoding)
    logits = outputs.logits

    # apply sigmoid + threshold
    sigmoid = Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(predicted_labels)


def parse_raw_arguments(raw_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_dir", type=Path, required=True,
        help="directory for model files"
    )
    parser.add_argument(
        "-t", "--text", required=True,
        help="input text for classification"
    )
    parser.add_argument("--loglevel", default="INFO", help="level for log output")
    args = parser.parse_args(raw_args)
    logging.basicConfig(
        format="{asctime} {levelname} {filename} -- {message}",
        style="{",
        level=getattr(logging, args.loglevel.upper()),
        handlers=[logging.StreamHandler()]
    )
    return args


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1:])
    logging.info("Finished finetuning the model, duration: {}".format(datetime.now() - start))