# encoding=utf-8
"""
  python3 test.py --tag your_tag --model your_model --load checkpoints/your_tag/500_model.pt --gpu 0

"""

import os
import pdb

import torch
import numpy as np
from PIL import Image

import dataloader as dl
from network import get_model
from options import opt
import misc_utils as utils

if not opt.load:
    raise Exception('Checkpoint must be specified at test phase, try --load <checkpoint_dir>')


Model = get_model(opt.model)
model = Model(opt)

model = model.to(device=opt.device)
model.eval()

load_epoch = model.load(opt.load)

result_dir = os.path.join(opt.result_dir, opt.tag, str(load_epoch))
utils.try_make_dir(result_dir)

for i, data in enumerate(dl.test_dataloader):
    utils.progress_bar(i, len(dl.test_dataloader), 'Test... ')
    img, paths = data['input'], data['path']
    img = img.to(device=opt.device)
    """
    Test Codes
    """
    filename = utils.get_file_name(paths[0])

    res = model.inference(img, progress_idx=(i, len(dl.test_dataloader)))

    # 保存结果
    save_path = os.path.join(result_dir, filename + '.png')

    Image.fromarray(res).save(save_path)