# encoding=utf-8
from configs.data_roots import get_one_dataset
from configs.transforms import get_transform
from options import opt, config
from options.helper import is_first_gpu

from dataloader.voc import VOCTrainValDataset
from dataloader.coco import CocoDataset

import torch.utils.data.dataset as dataset
import cv2

class ListTrainValDataset(dataset.Dataset):
    def __init__(self, *datasets, transforms=None):
        self.datasets = datasets
        self.transforms = transforms
        self.length = sum([len(dataset) for dataset in datasets])

    def __getitem__(self, index):
        total = 0
        for dataset in self.datasets:
            if index < total + len(dataset) :
                return dataset[index - total]

            total += len(dataset)

        raise IndexError('index out of range.')

    def __len__(self):
        return self.length


dataset_variables = ['voc_root', 'train_split', 'val_split', 'class_names', 'img_format', 'data_format']

class ListDatasetItem(object):
    def __init__(self, d):
        self.d = d

        for v in dataset_variables:
            # 等价于 exec(f'{v}=d.{v}')
            if hasattr(d, v):
                setattr(self, v, getattr(d, v))  # 把类的成员变量赋值给当前的局部变量

def get_all_datasets():
    if 'TRANSFORM' not in config.DATA:
        config.DATA.TRANSFORM = None

    transform = get_transform(config.DATA.TRANSFORM)
    t = transform(config)

    if hasattr(t, 'train_transform'):
        train_transform = t.train_transform

    if hasattr(t, 'val_transform'):
        val_transform = t.val_transform

    if isinstance(config.DATA.DATASET, str):
        config.DATA.DATASET = [config.DATA.DATASET]
    
    train_datasets = []
    val_datasets = []

    class_names = []
    for dataset_name in config.DATA.DATASET:
        dataset = get_one_dataset(dataset_name)
        d = dataset()
        dataset_item = ListDatasetItem(d)
        # class_names 是所有数据集name的并集
        for name in dataset_item.class_names:
            if name not in class_names:
                class_names.append(name)


        data_format = dataset_item.data_format
        if data_format == 'VOC':
            if hasattr(dataset_item, 'train_split'):
                train_dataset = VOCTrainValDataset(dataset_item.voc_root, 
                        dataset_item.class_names,
                        split=dataset_item.train_split,
                        format=dataset_item.img_format,
                        transforms=train_transform,
                        first_gpu=is_first_gpu())
                train_datasets.append(train_dataset)

            if hasattr(dataset_item, 'val_split'):
                val_dataset = VOCTrainValDataset(dataset_item.voc_root,
                        dataset_item.class_names,
                        split=dataset_item.val_split,
                        format=dataset_item.img_format,
                        transforms=val_transform,
                        first_gpu=is_first_gpu())
                val_datasets.append(val_dataset)

        elif data_format == 'COCO':
            if hasattr(dataset_item, 'train_split'):
                train_dataset = CocoDataset(dataset_item.voc_root, 
                        dataset_item.train_split, transforms=train_transform)
                train_datasets.append(train_dataset)

            if hasattr(dataset_item, 'val_split'):
                val_dataset = CocoDataset(dataset_item.voc_root, 
                        dataset_item.val_split, transforms=val_transform)
                val_datasets.append(val_dataset)
        
    config.DATA.CLASS_NAMES = class_names
    config.DATA.NUM_CLASSESS = len(class_names)

    if len(train_datasets):
        train_dataset = ListTrainValDataset(*train_datasets, transforms=train_transform)
    else:
        train_dataset = None

    if len(val_datasets):
        val_dataset = ListTrainValDataset(*val_datasets, transforms=val_transform)
    else:
        val_dataset = None

    return train_dataset, val_dataset
        

