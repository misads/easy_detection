import os
import random
import sys
from options import opt
from options import get_command_run
import misc_utils as utils
import warnings
import torch
import json
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def init_log(training=True):
    if training:
        log_dir = os.path.join(opt.log_dir, opt.tag)
    else:
        log_dir = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))

    utils.try_make_dir(log_dir)
    logger = utils.get_logger(f=os.path.join(log_dir, 'log.txt'), mode='a', level='info')

    logger.info('==================Options==================')
    for k, v in opt._get_kwargs():
        logger.info(str(k) + '=' + str(v))
    logger.info('===========================================')
    return logger

def get_gpu_id():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        gpu_id = str(gpu_id)
    else:
        gpu_id = str(opt.gpu_ids)

    return gpu_id

def load_meta(new=False):
    path = os.path.join(opt.log_dir, opt.tag, 'meta.json')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            meta = json.load(f)
    else:
        meta = []

    if new:
        new_meta = {
            'command': get_command_run(),
            'starttime': utils.get_time_stamp(),
            'best_acc': 0.,
            'gpu': get_gpu_id(),
            'opt': opt.__dict__,
        }
        meta.append(new_meta)
    return meta

def save_meta(meta):
    path = os.path.join(opt.log_dir, opt.tag, 'meta.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)

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
