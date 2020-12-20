# encoding=utf-8
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader.custom import get_dataset
from dataloader.transforms import get_transform

from dataloader.voc import VOCTrainValDataset
from dataloader.coco import CocoDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from options import opt
from mscv.image import tensor2im
import torch
import pdb

# 根据--opt.dataset获奖数据集
dataset = get_dataset(opt.dataset)
d = dataset()

if not opt.transform:
    opt.transform = opt.model.lower() if opt.model else 'none'

transform = get_transform(opt.transform)
t = transform()

dataset_variables = ['voc_root', 'train_split', 'val_split', 'class_names', 'img_format', 'data_format']

transform_variables = ['width', 'height', 'train_transform', 'val_transform']

for v in dataset_variables:
    # 等价于 exec(f'{v}=d.{v}')
    if hasattr(d, v):
        locals()[v] = getattr(d, v)  # 把类的成员变量赋值给当前的局部变量

for v in transform_variables:
    if hasattr(t, v):
        locals()[v] = getattr(t, v)  # 把类的成员变量赋值给当前的局部变量


opt.class_names = class_names
opt.num_classes = len(class_names)

opt.width = width
opt.height = height


def collate_fn(batch):
    target = {}
    b = len(batch)
    target['image'] = torch.stack([sample['image'] for sample in batch])
    target['bboxes'] = [sample['bboxes'] for sample in batch]
    target['labels'] = [sample['labels'] for sample in batch]
    target['path'] = [sample['path'] for sample in batch]
    target['yolo_boxes'] = torch.stack([sample['yolo_boxes'] for sample in batch])
    target['yolo4_boxes'] = torch.stack([sample['yolo4_boxes'] for sample in batch])
    target['yolo5_boxes'] = torch.cat(  # [b*50, 6] batch中第几张图片, label, c_x, c_y, w, h
                                [torch.cat([torch.ones([batch[i]['yolo5_boxes'].shape[0], 1]) * i,
                                batch[i]['yolo5_boxes']], 1) for i in range(b)], 0)
    return target

"""
Datasets
"""
if data_format == 'VOC':
    if hasattr(d, 'train_split'):
        train_dataset = VOCTrainValDataset(voc_root, 
                class_names,
                split=train_split,
                format=img_format,
                transforms=train_transform)

    if hasattr(d, 'val_split'):
        val_dataset = VOCTrainValDataset(voc_root,
                class_names,
                split=val_split,
                format=img_format,
                transforms=val_transform)

elif data_format == 'COCO':
    if hasattr(d, 'train_split'):
        train_dataset = CocoDataset(voc_root, train_split, transforms=train_transform)

    if hasattr(d, 'val_split'):
        val_dataset = CocoDataset(voc_root, val_split, transforms=val_transform)

"""
Dataloaders
"""
if hasattr(d, 'train_split'):
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        drop_last=True)
else:
    train_dataloader = None

if hasattr(d, 'val_split'):
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=opt.batch_size,
        num_workers=opt.workers // 2,
        drop_last=False)
else:
    val_dataloader = None