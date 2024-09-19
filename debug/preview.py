# encoding: utf-8
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import torch
import ipdb
import cv2
import numpy as np
from os.path import join
from options import opt, config
# from dataloader import paired_dataset
from mscv.summary import create_summary_writer, write_image
from mscv.image import tensor2im

from dataloader.dataloaders import train_dataloader, val_dataloader

from utils import visualize_boxes, denormalize_image
from misc_utils import progress_bar, get_file_name

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

"""
source domain 是clear的
"""
# writer = create_summary_writer('logs/preview')
vis_root = f'logs/preview/{opt.tag}'
os.makedirs(vis_root, exist_ok=True)
# class_names = config.DATA.CLASS_NAMES
class_names = config.DATA.CLASS_NAMES

preview = train_dataloader  # train_dataloader, val_dataloader

for i, sample in enumerate(preview):
    # if i > 30:
    #     break
    progress_bar(i, len(preview), 'Handling...')
    if opt.debug:
        ipdb.set_trace()

    ori_image = sample['ori_image']
    ori_sizes = sample['ori_sizes']
    paths = sample['path']
    
    image = sample['image'][0].detach().cpu().numpy().transpose([1,2,0])
    image = denormalize_image(image)
    image = (image.copy()*255).astype(np.uint8)

    bboxes = sample['bboxes'][0].cpu().numpy()
    labels = sample['labels'][0].cpu().numpy().astype(np.int32)

    visualize_boxes(image=image, boxes=bboxes, labels=labels, probs=np.array(np.random.randint(100, 101, size=[len(bboxes)])/100), class_labels=class_names)

    filename = get_file_name(paths[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # h, w = ori_sizes[0]
    # image = cv2.resize(image, (w, h))
    cv2.imwrite(join(vis_root, f'{filename}.jpg'), image)
    # write_image(writer, f'preview_{config.DATA.DATASET}/{i}', 'image', image, 0, 'HWC')

