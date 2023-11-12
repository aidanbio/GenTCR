import os.path

import torch
import transformers
from argparse import ArgumentParser
import warnings
from peft import AutoPeftModel
from transformers import AutoTokenizer

from tcredit.exp import Experiment
from tcredit.data import EpitopeTargetDataset
import logging
from transformers.utils import logging as hf_logging

# Logger
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger('tcredit')
warnings.filterwarnings('ignore')


def generate_datasets(args):
    logger.info(f'Start generate_datasets with args: {args}')

    EpitopeTargetDataset.FN_DATA_CONFIG = args.data_config
    for data_key in args.data.split(','):
        logger.info(f'Generating dataset for {data_key}')
        try:
            ds = EpitopeTargetDataset.from_key(data_key, args=args)
            logger.info(f'Done to generate data for {data_key}, the number of data: {len(ds)}')
        except Exception as e:
            logger.error(f'Failed to generate data for {data_key}: {e}')


def run_task(args):
    logger.info(f'Start run_task with {args}...')
    exp = Experiment.from_key(args.exp)
    task = exp.create_task(args.task)
    task.run(args)


def save_task_bm(args):
    logger.info(f'Start save_task_bm with {args}...')
    exp = Experiment.from_key(args.exp)
    task = exp.create_task(args.task)
    output_dir = task.config['output_dir']
    bm_path = os.path.join(output_dir, args.bm_name)
    logger.info(f'Loading best model from {bm_path}')
    model = AutoPeftModel.from_pretrained(bm_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    logger.info(f'Saving best model to {output_dir}')
    model.save_pretrained(output_dir, safe_serialization=True)
    logger.info(f'Saving tokenizer to {output_dir}')
    tokenizer.save_pretrained(output_dir)

def main():
    parser = ArgumentParser('tcredit')
    parser.add_argument('--log_level', type=str, default='INFO')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'generate_datasets'
    sub_parser = subparsers.add_parser('generate_datasets')
    sub_parser.set_defaults(func=generate_datasets)
    sub_parser.add_argument('--data_config', type=str, default='../config/data.json')
    sub_parser.add_argument('--data', type=str, default='immunecode')
    sub_parser.add_argument('--n_workers', type=int, default=1)

    # Arguments for sub command 'run_task'
    sub_parser = subparsers.add_parser('run_task')
    sub_parser.set_defaults(func=run_task)
    sub_parser.add_argument('--exp', type=str, default='testexp')
    sub_parser.add_argument('--task', type=str, default='mlm_finetune')

    # Arguments for sub command 'save_task_bm'
    sub_parser = subparsers.add_parser('save_task_bm')
    sub_parser.set_defaults(func=save_task_bm)
    sub_parser.add_argument('--exp', type=str, default='testexp')
    sub_parser.add_argument('--task', type=str, default='mlm_finetune')
    sub_parser.add_argument('--bm_name', type=str, default='checkpoint-3')

    args = parser.parse_args()

    print(f'Logging level: {args.log_level}')
    logger.setLevel(args.log_level)
    hf_logging.set_verbosity(args.log_level)

    args.func(args)


if __name__ == '__main__':
    main()
