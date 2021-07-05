import os
import random
import sys
import misc_utils as utils
import warnings
import torch
import json
import yaml
import numpy as np
from easydict import EasyDict

from itertools import repeat
import collections.abc

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def get_command_run():
    args = sys.argv.copy()
    args[0] = args[0].split('/')[-1]

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        command = f'CUDA_VISIBLE_DEVICES={gpu_id} '
    else:
        command = ''

    if sys.version[0] == '3':
        command += 'python3'
    else:
        command += 'python'

    for i in args:
        command += ' ' + i
    return command

    
def parse_config(yml_path):
    if not os.path.isfile(yml_path):
        raise_exception(f'{yml_path} not exists.')

    with open(yml_path, 'r') as f:
        try:
            configs = yaml.safe_load(f.read())
        except yaml.YAMLError:
            raise_exception('Error parsing YAML file:' + path)

    default_configs = EasyDict({
        'MODEL':{
            'BACKBONE': None
        },
        'MISC':{
            'VAL_FREQ': 1,
            'SAVE_FREQ': 1,
            'LOG_FREQ': 1,
            'NUM_WORKERS': 4
        },
        'TEST':{
            'NMS_THRESH': 0.5,
            'CONF_THRESH': 0.05
        }
    })
    default_configs.update(configs)
    return default_configs


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def raise_exception(msg, error_code=1):
    utils.color_print('Exception: ' + msg, 1)
    exit(error_code)


def deprecated(info=''):
    def decorator(fn):
        def deprecation_info(*args, **kwargs):
            warnings.warn(info, DeprecationWarning)
            utils.color_print(f'DeprecationWarning: {info}', 1)
            result = fn(*args, **kwargs)
            return result

        return deprecation_info
    return decorator
