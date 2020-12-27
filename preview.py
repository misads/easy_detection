# encoding: utf-8
import torch
import ipdb
import cv2
import numpy as np
from options import opt
# from dataloader import paired_dataset
from mscv.summary import create_summary_writer, write_image
from mscv.image import tensor2im

from dataloader.dataloaders import train_dataloader, val_dataloader
from dataloader import voc

from utils.vis import visualize_boxes

import misc_utils as utils

import random
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

"""
source domain 是clear的
"""
writer = create_summary_writer('logs/preview')

# class_names = opt.class_names
class_names = opt.class_names

preview = train_dataloader  # train_dataloader, val_dataloader

for i, sample in enumerate(preview):
    # if i > 30:
    #     break
    utils.progress_bar(i, len(preview), 'Handling...')
    if opt.debug:
        ipdb.set_trace()

    image = sample['image'][0].detach().cpu().numpy().transpose([1,2,0])
    image = (image.copy()*255).astype(np.uint8)

    bboxes = sample['bboxes'][0].cpu().numpy()
    labels = sample['labels'][0].cpu().numpy().astype(np.int32)

    visualize_boxes(image=image, boxes=bboxes, labels=labels, probs=np.array(np.random.randint(100, 101, size=[len(bboxes)])/100), class_labels=class_names)

    write_image(writer, f'preview_{opt.dataset}/{i}', 'image', image, 0, 'HWC')

writer.flush()
