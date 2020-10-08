# encoding=utf-8
import ipdb

import torch
import torchvision.transforms.functional as F
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms
from options import opt
import albumentations as A
from mscv.summary import create_summary_writer, write_image
import dataloader.dataloaders
from dataloader.voc import VOCTrainValDataset

import xml.etree.ElementTree as ET
import misc_utils as utils
import random
import numpy as np
import cv2


preview_transform = A.Compose(
    [  # Empty
    ], 
    p=1.0, 
    bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0, 
        min_visibility=0,
        label_fields=['labels']
    )
)

colors = [(0.,1.,0.), (0.,0.,1.), (1.,0.,0.), (0.,1.,1.), (1.,0.,1.), (1.,1.,0.), (1.,1.,1.), (0.,0.,0.), (.5,0.,0.), 
(0.,.5,0.), (0.,0.,.5), (0.,.5,.5), (.5,0.,.5), (.5,.5,0.), (.5,.5,.5), (.5,1.,0.), (.5,0.,1.), (1.,.5,0.), 
(0.,.5,1.), (1.,0.,.5), (0.,1.,.5), (.5,.5,1.), (.5,1.,.5), (1.,.5,.5), (1.,1.,.5), (1.,.5,1.), (.5,1.,1.)]


if __name__ == '__main__':
    voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    cityscapes_classes = ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']
    apollo_classes = ['bicycle', 'bicycle_group', 'bus', 'car', 'car_groups', 'motorbicycle',
    'motorbicycle_group', 'person', 'person_group', 'rider', 'rider_group', 'tricycle', 'truck']

    opt.dataset = opt.dataset.lower()

    datasets = {'voc': ['VOC', voc_classes, 'jpg'],
                'cityscapes': ['cityscapes', cityscapes_classes, 'png'],
                'wheat': ['wheat_detection', ['wheat'], 'jpg'],
                'apollo': ['apollo', apollo_classes, 'png'],
                'widerface': ['wider_face', ['face'], 'jpg'],
               }

    root_path, class_names, format = datasets[opt.dataset]
    root_path = os.path.join('datasets', root_path)

    writer = create_summary_writer('logs/preview_voc')
    voc_dataset = VOCTrainValDataset(root_path, class_names, 'train.txt', format, transforms=preview_transform)

    for i, sample in enumerate(voc_dataset):
        if i >= 30:
            break

        utils.progress_bar(i, len(voc_dataset), 'Drawing...')

        image = sample['image']
        bboxes = sample['bboxes']
        labels = sample['labels'] 
        image_path = sample['path']

        image = image.copy()

        for j in range(len(labels)):
            x1, y1, x2, y2 = bboxes[j]
            label = int(labels[j].item())
            label_name = class_names[label]

            cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 2)
            cv2.putText(image, label_name, (x1, y1-3), 0, 1, colors[label], 2)

        write_image(writer, f'preview/{i}', 'image', image, 0, 'HWC')

    writer.flush()
    print('Done. Run `tensorboard --logdir logs/preview_voc` to vis the result')
