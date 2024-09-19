import os
import random
import sys
import warnings
import torch
import json
import yaml
import numpy as np

from itertools import repeat
from misc_utils import color_print
import collections.abc


class SumMeters:
    """AverageMeter class
    Example
        >>> avg_meters = AverageMeters()
        >>> for i in range(100):
        >>>     avg_meters.update({'f': i})
        >>>     print(str(avg_meters))
    """

    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}

    def update(self, new_dic, weight=1.):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key] * weight
            else:
                self.dic[key] += new_dic[key] * weight

    def __getitem__(self, key):
        return self.dic[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()

    def items(self):
        return self.dic.items()


class AverageMeters:
    """AverageMeter class
    Example
        >>> avg_meters = AverageMeters()
        >>> for i in range(100):
        >>>     avg_meters.update({'f': i})
        >>>     print(str(avg_meters))
    """

    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic, weight=1.):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key] * weight
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key] * weight
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()

    def items(self):
        return self.dic.items()
        

class EasyDict:
    def __init__(self, data: dict):
        self._dict = data

    def update(self, data):
        self._dict.update(data)

    def __iter__(self):
        return self._dict.__iter__()

    def __setattr__(self, attrname, value):
        if attrname == '_dict':
            return super(EasyDict, self).__setattr__(attrname, value)

        self._dict[attrname] = value

    def __getattr__(self, attrname):
        if attrname in self._dict:
            attvalue = self._dict[attrname]
            if isinstance(attvalue, dict):
                return EasyDict(attvalue)
            else:
                return attvalue

        return None

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __repr__(self):
        return str(self._dict)


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
        exception(f'{yml_path} not exists.')

    with open(yml_path, 'r') as f:
        try:
            configs = yaml.safe_load(f.read())
        except yaml.YAMLError:
            exception('Error parsing YAML file:' + yml_path)

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


def exception(msg, error_code=1):
    color_print('Exception: ' + msg, 1)
    exit(error_code)


def deprecated(info=''):
    def decorator(fn):
        def deprecation_info(*args, **kwargs):
            warnings.warn(info, DeprecationWarning)
            color_print(f'DeprecationWarning: {info}', 1)
            result = fn(*args, **kwargs)
            return result

        return deprecation_info
    return decorator


def warning(info=''):
    def decorator(fn):
        def warning_info(*args, **kwargs):
            warnings.warn(info, Warning)
            color_print(f'Warning: {info}', 1)
            result = fn(*args, **kwargs)
            return result

        return warning_info
    return decorator


def denormalize_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0):
    """
    Args:
        img: np.ndarray (h, w, 3), 
    """
    if np.min(img) > -0.01 and np.max(img) < 1.01:  # 不需要denorm
        return img

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    img = img.astype(np.float32)
    img *= std
    img += mean

    img[img>1.] = 1.
    img[img<0.] = 0.
    return img
    