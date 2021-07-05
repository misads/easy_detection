# encoding=utf-8
import ipdb

import torch
import torchvision.transforms.functional as F
import os
import json
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms
from options import opt
import albumentations as A
from mscv.summary import create_summary_writer, write_image
import dataloader.dataloaders
from dataloader.voc import VOCTrainValDataset
from configs.data_roots import get_dataset

import xml.etree.ElementTree as ET
import misc_utils as utils
import random
import numpy as np
import cv2


preview_transform = A.Compose(
    [  # 没有任何变换
    ], 
    p=1.0, 
    bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0, 
        min_visibility=0,
        label_fields=['labels']
    )
)

coco_dataset = {'images': [], 'annotations': [], 'categories': []}
global_image_id = 0
global_annotation_id = 0

if __name__ == '__main__':
    opt.dataset = 'voc'
    dataset = get_dataset(opt.dataset)
    d = dataset()

    variable_names = ['voc_root', 'train_split', 'val_split', 'class_names', 'img_format', 
                      'width', 'height', 'train_transform', 'val_transform']

    for v in variable_names:
        # 等价于 exec(f'{v}=d.{v}')
        locals()[v] = getattr(d, v)  # 把类的成员变量赋值给当前的局部变量

    config.DATA.CLASS_NAMES = class_names
    config.DATA.NUM_CLASSESS = len(class_names)

    for global_category_id, class_name in enumerate(class_names, 1):
        coco_dataset['categories'].append({   
            'supercategory': 'class',
            'id': global_category_id,
            'name': class_name
        })


    opt.width = width
    opt.height = height

    voc_dataset = VOCTrainValDataset(voc_root, 
            class_names,
            split=train_split,
            format=img_format,
            transforms=preview_transform)

    output_file = f'instances_{train_split[:-4]}.json'

    for i, sample in enumerate(voc_dataset):
        utils.progress_bar(i, len(voc_dataset), 'Drawing...')

        image = sample['image']
        bboxes = sample['bboxes'].cpu().numpy()
        labels = sample['labels'].cpu().numpy()
        image_path = sample['path']
        
        h, w, _ = image.shape

        global_image_id += 1

        coco_dataset['images'].append({   
            'file_name': os.path.basename(image_path),
            'id': global_image_id,
            'width': int(w),
            'height': int(h)
        })

        for j in range(len(labels)):
            x1, y1, x2, y2 = bboxes[j]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2),  
            category_id = int(labels[j].item()) + 1
            # label_name = class_names[label]

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            area = width * height

            global_annotation_id += 1

            coco_dataset['annotations'].append({
                'id': global_annotation_id,
                'image_id': global_image_id,
                'category_id': category_id,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                'area': float(area),
                'iscrowd': 0,
                'bbox': [x1, y1, width, height],

            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_dataset, f, ensure_ascii=False)

    print(f'Done. coco json file has been saved to `{output_file}`')