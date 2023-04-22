#!/usr/bin/env python3


import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
#from json import dump

import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


class Preprocessor:

    def __init__(self, tokenizer: AutoTokenizer, labels: List[str]):
        self.tokenizer = tokenizer
        self.labels = labels

    def process_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        text = batch["Tweet"]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128)
        labels_batch = {k: batch[k] for k in batch.keys() if k in self.labels}
        labels_matrix = np.zeros((len(text), len(self.labels)))
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()  
        return encoding


def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average = "micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {"f1": f1_micro_average,
               "roc_auc": roc_auc,
               "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def main(raw_args: List[str]):
    args = parse_raw_arguments(raw_args)

    logging.info(f"loading data for finetuning from : {args.data_dir}")
    dataset = load_from_disk(args.data_dir)
    labels = [label for label in dataset["train"].features.keys() if label not in ["ID", "Tweet"]]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    logging.info(f"found labels: {labels}")

    logging.info(f"processing data sets, tokanizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    preprocessor = Preprocessor(tokenizer, labels)
    encoded_dataset = dataset.map(preprocessor.process_batch, batched=True, remove_columns=dataset["train"].column_names)
    encoded_dataset.set_format("torch")

    logging.info(f"loading pretrained model: {args.base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id,
                                                           cache_dir=args.cache_dir)
    logging.info("setup finetuning parameters")
    train_args = TrainingArguments(
        args.model_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        no_cuda=not args.use_gpu
        #push_to_hub=True,
    )
    trainer = Trainer(
        model,
        train_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logging.info("finetuning the model")
    trainer.train(resume_from_checkpoint=False)
    trainer.evaluate()
    logging.info(f"saving model at: {args.model_dir}")
    trainer.save_model(args.model_dir)
    with args.model_dir.joinpath("id2label.json").open("w") as file:
        dump(id2label, file)



def parse_raw_arguments(raw_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", type=Path, required=True,
        help="directory with data files"
    )
    parser.add_argument(
        "-m", "--model_dir", type=Path, required=True,
        help="directory for model files"
    )
    parser.add_argument(
        "-c", "--cache_dir", type=Path, default=Path("~/.cache/huggingface/hub"),
        help="directory for model files"
    )
    parser.add_argument(
        "--base_model", type=str, default="distilbert-base-uncased",
        help="Name of the pretrained model to use (default=distilbert-base-uncased)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="size of each batch for finetuninng (default=8)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1,
        help="number of epchs to train (default=1)"
    )
    parser.add_argument(
        "--use_gpu", action="store_true",
        help="if GPU should be used during finetuning"
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