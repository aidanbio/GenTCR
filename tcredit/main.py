import transformers
from argparse import ArgumentParser
import warnings
from tcredit.exp import Experiment
from tcredit.data import EpitopeTargetDataset
import logging
from transformers.utils import logging as hf_logging

# Logger
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger(__name__)
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

    args = parser.parse_args()

    print(f'Logging level: {args.log_level}')
    logger.setLevel(args.log_level)
    hf_logging.set_verbosity(args.log_level)

    args.func(args)


if __name__ == '__main__':
    main()
