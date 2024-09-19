import argparse
import json
import sys
import os

import torch

from misc_utils import get_file_name, get_time_str
from utils import parse_config, exception, get_command_run

"""
    Arg parse
    opt = parse_args()
"""


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default=None,
                        help='指定保存结果的路径, 如果未指定, 结果会保存在与配置文件文件名相同路径')

    parser.add_argument('--config', type=str, default=None, help='(必须指定) yml配置文件')

    parser.add_argument('--gpu_id', '--gpu', type=int, default=0, help='gpu id: e.g. 0 . use -1 for CPU')
    parser.add_argument("--local_rank", type=int, default=None, help='only used in dist train mode')
    
    # # training options
    parser.add_argument('--debug', action='store_true', help='debug模式')
    parser.add_argument('--vis', action='store_true', help='可视化测试结果')
    parser.add_argument('--gt', action='store_true', help='可视化gt, 仅在--vis 时有效')

    parser.add_argument('--load', type=str, default=None, help='指定载入checkpoint的路径')
    parser.add_argument('--resume', action='store_true', help='resume training, 仅在指定--load时有效')

    # parser.add_argument('--seed', type=int, default=None, help='random seed')

    # # test settings
    parser.add_argument('--no_val', '-no_eval', action='store_true', help='--no_val训练时不会进行验证')

    return parser.parse_args()


def set_config():
    if opt.config:
        config = parse_config(opt.config)
    else:
        exception('--config must be specified.')

    if isinstance(config.DATA.SCALE, int):
        config.DATA.SCALE = (config.DATA.SCALE, config.DATA.SCALE)  # make tuple

    if not opt.tag:
        opt.tag = get_file_name(opt.config)


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
            f.writelines(get_time_str(fmt="%Y-%m-%d %H:%M:%S") + ' ' + pid + ' ' + get_command_run() + '\n')

    if config.TRAIN is None:
        config.TRAIN = {}

    if config.TRAIN.ROI is None:
        config.TRAIN.ROI = {}

    if config.TRAIN.RPN is None:
        config.TRAIN.RPN = {}

    if config.TRAIN.FPN is None:
        config.TRAIN.FPN = {}

    if config.TEST is None:
        config.TEST = {}

    if config.TEST.RPN is None:
        config.TEST.RPN = {}
        
    return config


opt = parse_args()
config = set_config()
