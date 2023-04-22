#!/usr/bin/env python3


import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

from datasets import load_dataset


def main(raw_args: List[str]):
    args = parse_raw_arguments(raw_args)
    logging.info("getting dataset from huggingface: sem_eval_2018_task_1, subtask5.english")
    dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")
    logging.info(f"saving to : {args.out_dir}")
    dataset.save_to_disk(args.out_dir)


def parse_raw_arguments(raw_args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--out_dir', type=Path, required=True,
        help='directory for output files'
    )
    parser.add_argument('--loglevel', default='INFO', help='level for log output')
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
    logging.info('Finished creating data, duration: {}'.format(datetime.now() - start))