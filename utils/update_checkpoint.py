# encoding=utf-8
"""
    将旧的checkpoint升级成新的checkpoint
"""

import argparse
import torch
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='input checkpoint')
    parser.add_argument('output', type=str, help='output checkpoint')
    parser.add_argument('--epoch', type=int, default=0)
    return parser.parse_args()


opt = parse_args()


if not os.path.isfile(opt.input):
    raise FileNotFoundError

state_dict = torch.load(opt.input, map_location='cpu')

save_dict = {
    'cleaner': state_dict,
    'epoch': opt.epoch
}

torch.save(save_dict, opt.output)

