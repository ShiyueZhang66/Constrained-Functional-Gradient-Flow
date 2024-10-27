import argparse
import logging
import os
import shutil
import numpy as np
import torch
import yaml
import time
from runners import ConstrainedGWG, PrimalDualSVGD, ControlledSVGD, MIED


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Path to the config file')
    parser.add_argument('--runner',
                        type=str,
                        required=True,
                        help='Runner')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--run',
                        type=str,
                        default='run',
                        help='Path for saving running related data.')
    parser.add_argument('--doc',
                        type=str,
                        default='0',
                        help='A string for documentation purpose')
    parser.add_argument('--comment',
                        type=str,
                        default='',
                        help='A string for experiment comment')
    parser.add_argument('--load_path',
                        type=str,
                        default='',
                        help='Path for loading models')

    args = parser.parse_args()

    args.log = os.path.join(args.run, args.doc)

    # parse config file
    if os.path.exists(args.log):
        shutil.rmtree(args.log)
    os.makedirs(args.log)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.path = args.log
    new_config.load_path = args.load_path

    with open(os.path.join(args.log, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # setup logger
    level = getattr(logging, 'INFO', None)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info('Using device: {}'.format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == '__main__':
    args, config = parse_args_and_config()
    logging.info('Writing log file to {}'.format(args.log))
    logging.info('Exp instance id = {}'.format(os.getpid()))
    logging.info('Exp comment = {}'.format(args.comment))
    logging.info('Random seed = {}'.format(args.seed))
    logging.info('Config =')
    print('>' * 80)
    print(config)
    print('<' * 80)

    if args.runner == "constrained_gwg":
        runner = ConstrainedGWG(config)
    elif args.runner == "controlled_svgd":
        runner = ControlledSVGD(config)
    elif args.runner == 'pd_svgd':
        runner = PrimalDualSVGD(config)
    elif args.runner == 'mied':
        runner = MIED(config)

    start_time = time.perf_counter()
    runner.train()
    runtime = time.perf_counter() - start_time
    logging.info("runtime: {:.1f}".format(runtime))
