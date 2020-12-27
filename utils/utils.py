import os
import random
import sys
from options import opt
import misc_utils as utils
import warnings
import torch
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
