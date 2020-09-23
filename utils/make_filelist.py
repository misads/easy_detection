# encoding=utf-8
"""
    生成列表数据集
    python make_filelist.py --input yourdata/input --label yourdata/label --val_ratio 0.2
"""

import random
import argparse
import torch
import os
import misc_utils as utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.)
    parser.add_argument('--out', type=str, default='datasets')
    return parser.parse_args()


opt = parse_args()


if not os.path.isdir(opt.input):
    raise FileNotFoundError

os.makedirs(opt.out, exist_ok=True)

files = os.listdir(opt.input)


def is_image(file_name):
    return not file_name.startswith('.') and ('jpg' in file_name or 'png' in file_name)


files = list(filter(is_image, files))


if opt.label is None:  # test
    files.sort()
    test_count = len(files)
    with open(os.path.join(opt.out, 'test.txt'), 'w') as f:
        for line in files:
            line = os.path.join(os.path.abspath(opt.input), line)
            print(line)
            f.writelines(line + '\n')
        utils.color_print(f'test count: {test_count}', 3)

else:
    random.shuffle(files)
    count = len(files)
    val_count = int(count*opt.val_ratio)
    train_count = count - val_count
    val = random.sample(files, val_count)
    train = []
    for file in files:
        if file not in val:
            train.append(file)

    with open(os.path.join(opt.out, 'train.txt'), 'w') as f:
        for line in train:
            line = os.path.join(os.path.abspath(opt.input), line) + ' ' + os.path.join(os.path.abspath(opt.label), line)
            print(line)
            f.writelines(line + '\n')

    utils.color_print(f'train count: {train_count}', 3)

    if val_count > 0:
        with open(os.path.join(opt.out, 'val.txt'), 'w') as f:
            for line in val:
                line = os.path.join(os.path.abspath(opt.input), line) + ' ' + os.path.join(os.path.abspath(opt.label), line)
                print(line)
                f.writelines(line + '\n')
        utils.color_print(f'val count: {val_count}', 3)



