# encoding=utf-8
"""
pushlist checkpoint
Usage:
    python3 utils/publish_checkpoint.py checkpoints/12_Faster_RCNN.pt

"""

from misc_utils import get_file_name

import argparse
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, help='input checkpoint')
    parser.add_argument('--out', type=str, default=None ,help='output path')
    return parser.parse_args()


opt = parse_args()


if not os.path.isfile(opt.checkpoint):
    raise FileNotFoundError

state_dict = torch.load(opt.checkpoint, map_location='cpu')

save_dict = {
    'detector': state_dict['detector'],
    'epoch': 0
}

if opt.out == None:
    filaname = get_file_name(opt.checkpoint)
    save_dir = 'checkpoints/publish'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{filaname}_publish.pt')
else:
    os.makedirs(os.path.dirname(opt.out))
    out_path = opt.out

torch.save(save_dict, out_path)
print(f'Saved to "{out_path}".')

