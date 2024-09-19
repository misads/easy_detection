import sys
import os
import json
import logging
from misc_utils import get_logger, get_time_stamp
from utils import get_command_run
from .options import opt, config


def init_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(f=os.path.join(log_dir, 'log.txt'), mode='a', level='info')
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)
    logger.info('==================Configs==================')
    with open(opt.config) as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            pos = line.find('#')
            if pos != -1:
                line = line[:pos]
            if len(line.strip()):
                logger.info(line)
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
        gpu_id = str(opt.gpu_id)

    return gpu_id

def load_meta(new=False):
    path = os.path.join('logs', opt.tag, 'meta.json')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            meta = json.load(f)
    else:
        meta = []

    if new:
        new_meta = {
            'command': get_command_run(),
            'starttime': get_time_stamp(),
            'best_acc': 0.,
            'gpu': get_gpu_id(),
            'opt': opt.__dict__,
            'config': config.__dict__
        }
        meta.append(new_meta)
    return meta

def save_meta(meta):
    path = os.path.join('logs', opt.tag, 'meta.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)

def is_first_gpu():
    # used in distributed mode
    return not opt.local_rank

def is_distributed():
    return opt.local_rank is not None


# 设置多卡训练
def setup_multi_processes():
    import torch.distributed as dist
    import torch
    import cv2
    import os

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if 'OMP_NUM_THREADS' not in os.environ:
        omp_num_threads = 1
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ:
        mkl_num_threads = 1
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)
