import argparse
import json
import sys
import os

import torch

import misc_utils as utils
from utils import parse_config, raise_exception, get_command_run

"""
    Arg parse
    opt = parse_args()
"""


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default=None,
                        help='folder name to save the outputs')

    parser.add_argument('--config', type=str, default=None, help='yml config file')

    parser.add_argument('--gpu_id', '--gpu', type=int, default=0, help='gpu id: e.g. 0 . use -1 for CPU')
    parser.add_argument("--local_rank", type=int, default=None, help='only used in dist train mode')
    
    # # training options
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--vis', action='store_true', help='vis eval result')

    parser.add_argument('--load', type=str, default=None, help='load checkpoint')
    # parser.add_argument('--weights', type=str, default=None, help='load checkpoint for Detector')

    parser.add_argument('--resume', action='store_true', help='resume training, only works with --load')
    # parser.add_argument('--reset', action='store_true', help='reset training, only used when --load')

    parser.add_argument('--save_path', '--save', type=str, default=None, help='save result path')
    # parser.add_argument('--seed', type=int, default=None, help='random seed')

    # # test time bbox settings
    parser.add_argument('--no_val', '-no_eval', action='store_true', help='do not evaluate')

    return parser.parse_args()


def set_config():
    if opt.config:
        config = parse_config(opt.config)
    else:
        raise_exception('--config must be specified.')

    if isinstance(config.DATA.SCALE, int):
        config.DATA.SCALE = (config.DATA.SCALE, config.DATA.SCALE)  # make tuple

    if not opt.tag:
        opt.tag = utils.get_file_name(opt.config)


    if opt.local_rank is not None:
        opt.gpu_id = opt.local_rank

    opt.device = 'cuda:' + str(opt.gpu_id) if torch.cuda.is_available() and opt.gpu_id != -1 else 'cpu'

    if opt.debug:
        config.MISC.SAVE_FREQ = 1
        config.MISC.VAL_FREQ = 1
        config.MISC.LOG_FREQ = 1

    if opt.tag != 'default':
        pid = f'[PID:{os.getpid()}]'
        with open('run_log.txt', 'a') as f:
            f.writelines(utils.get_time_str(fmt="%Y-%m-%d %H:%M:%S") + ' ' + pid + ' ' + get_command_run() + '\n')

    return config


opt = parse_args()
config = set_config()
