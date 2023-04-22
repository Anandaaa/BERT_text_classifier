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

INFERENCE_DIR = Path(__file__).resolve().parent
ROOT_PATH = INFERENCE_DIR.parent
with INFERENCE_DIR.joinpath("config.json").open("r") as file:
    INFERENCE_CONFIG = load(file)
MODEL_DIR = ROOT_PATH.joinpath("models", INFERENCE_CONFIG["model"])
with MODEL_DIR.joinpath("config.json").open("r") as file:
    MODEL_CONFIG = load(file)
ID2LABEL = {int(k): v for k, v in MODEL_CONFIG["id2label"].items()}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
TOKENIZER_PATH = MODEL_DIR.joinpath("tokenizer.json")
TOKANIZER = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
MODEL = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        problem_type="multi_label_classification",
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )


def classify(text: str) -> Dict[str, float]:

    # tokanize input
    encoding = TOKANIZER(text, return_tensors="pt")
    encoding = {k: v.to(MODEL.device) for k, v in encoding.items() if k in ["input_ids", "attention_mask"]}
    print(encoding)

    # apply model
    outputs = MODEL(**encoding)
    logits = outputs.logits

    # apply sigmoid + threshold
    sigmoid = Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1

    # turn predicted id's into actual label names
    #predicted_labels = [ID2LABEL[idx] for idx, label in enumerate(predictions) if label == 1.0]
    label_scores = {ID2LABEL[idx]: float(label) for idx, label in enumerate(probs)}
    return label_scores


def main(raw_args: List[str]):
    args = parse_raw_arguments(raw_args)
    label_scores = classify(args.text)
    print("label_scores:")
    for k, v in label_scores.items():
        print(f"{k}: {v}")


def parse_raw_arguments(raw_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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