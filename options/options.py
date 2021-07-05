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

    parser.add_argument('--gpu_ids', '--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    # parser.add_argument('--model', type=str, default=None, help='which model to use')
    # parser.add_argument('--backbone', type=str, default=None, help='which backbone to use')
    # parser.add_argument('--norm', type=str, choices=['batch', 'instance', None], default=None,
    #                     help='[instance] normalization or [batch] normalization')

    # # batch size
    # parser.add_argument('--batch_size', '-b', type=int, default=1, help='input batch size')

    # # optimizer and scheduler
    # parser.add_argument('--optimizer', choices=['adam', 'sgd', 'radam', 'lookahead', 'ranger'], default='adam')
    # parser.add_argument('--scheduler', default='1x')

    # # data augmentation
    # # parser.add_argument('--aug', action='store_true', help='Randomly scale, jitter, change hue, saturation and brightness')

    # # scale
    # parser.add_argument('--scale', type=int, default=None, help='scale images to this size')
    # parser.add_argument('--crop', type=int, default=None, help='then crop to this size')
    # parser.add_argument('--workers', '-w', type=int, default=4, help='num of workers')

    # # for datasets
    # parser.add_argument('--dataset', default='voc', help='training dataset')
    # parser.add_argument('--transform', default=None, help='transform')
    # parser.add_argument('--val_set', type=str, default=None)
    # parser.add_argument('--test_set', type=str, default=None)

    # # init weights
    # parser.add_argument('--init', type=str, default=None, help='{normal, xavier, kaiming, orthogonal}')

    # # training options
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--vis', action='store_true', help='vis eval result')

    parser.add_argument('--load', type=str, default=None, help='load checkpoint')
    # parser.add_argument('--weights', type=str, default=None, help='load checkpoint for Detector')

    parser.add_argument('--resume', action='store_true', help='resume training, only used when --load')
    # parser.add_argument('--reset', action='store_true', help='reset training, only used when --load')

    parser.add_argument('--save_path', '--save', type=str, default=None, help='save result path')
    # parser.add_argument('--epochs', '--max_epoch', type=int, default=500, help='epochs to train')
    # parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    # parser.add_argument('--seed', type=int, default=None, help='random seed')

    # # test time bbox settings
    # parser.add_argument('--conf_thresh', type=float, default=0.01, help='bboxes with conf < this threshold will be ignored')
    # parser.add_argument('--nms_thresh', type=float, default=0.45, help='nms threshold')
    # parser.add_argument('--wbf_thresh', type=float, default=0.5, help='wbf threshold')
    # parser.add_argument('--box_fusion', choices=['nms', 'wbf'], default='nms')

    # parser.add_argument('--save_freq', type=int, default=10, help='freq to save models')
    # parser.add_argument('--eval_freq', '--val_freq', type=int, default=10, help='freq to eval models')
    # parser.add_argument('--log_freq', type=int, default=1, help='freq to vis in tensorboard')
    parser.add_argument('--no_val', action='store_true', help='不要eval')

    # # test options
    # parser.add_argument('--tta', action='store_true', help='test with augmentation')
    # parser.add_argument('--tta-x8', action='store_true', help='test with augmentation x8')

    return parser.parse_args()


def set_config():
    if opt.config:
        config = parse_config(opt.config)
    else:
        raise_exception('--config must be specified.')

    if not opt.tag:
        opt.tag = utils.get_file_name(opt.config)

    opt.device = 'cuda:' + opt.gpu_ids if torch.cuda.is_available() and opt.gpu_ids != '-1' else 'cpu'

    if opt.debug:
        config.MISC.SAVE_FREQ = 1
        config.MISC.EVAL_FREQ = 1
        config.MISC.LOG_FREQ = 1

    if opt.tag != 'default':
        pid = f'[PID:{os.getpid()}]'
        with open('run_log.txt', 'a') as f:
            f.writelines(utils.get_time_str(fmt="%Y-%m-%d %H:%M:%S") + ' ' + pid + ' ' + get_command_run() + '\n')

    return config


opt = parse_args()
config = set_config()
