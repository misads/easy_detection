# encoding=utf-8
from dataloader.image_folder import FolderTrainValDataset, FolderTestDataset
from dataloader.image_list import ListTrainValDataset, ListTestDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader import yolo_dataset
from dataloader import paired_dataset
from dataloader.voc import VOCTrainValDataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from options import opt
from mscv.image import tensor2im
import torch
import pdb


###################

TEST_DATASET_HAS_OPEN = False  # 有没有开放测试集
DATA_FOTMAT = 'VOC'  # 数据集格式

###################



if DATA_FOTMAT == 'VOC':

    voc_root = 'datasets/wheat_detection'
    train_split = 'train.txt'
    val_split = 'val.txt' 
    class_names = ['wheat']
    opt.num_classes = len(class_names)

    train_transform = train_transform = A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                        val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                            contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=5, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        ),
    )


    voc_train_dataset = VOCTrainValDataset(voc_root, 
            class_names,
            split=train_split,
            transforms=train_transform)

    def collate_fn(batch):
        target = {}
        b = len(batch)
        target['image'] = torch.stack([sample['image'] for sample in batch])
        target['bboxes'] = [sample['bboxes'] for sample in batch]
        target['labels'] = [sample['labels'] for sample in batch]
        target['path'] = [sample['path'] for sample in batch]
        target['yolo_boxes'] = torch.stack([sample['yolo_boxes'] for sample in batch])
        target['yolo5_boxes'] = torch.cat(  # [b*50, 6] batch中第几张图片, label, c_x, c_y, w, h
                                    [torch.cat([torch.ones([batch[i]['yolo5_boxes'].shape[0], 1]) * i,
                                    batch[i]['yolo5_boxes']], 1) for i in range(b)], 0)
        return target

    voc_train_dataloader = torch.utils.data.DataLoader(voc_train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=opt.batch_size,
        num_workers=4,
        drop_last=True)

    val_transform = A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

    voc_val_dataset = VOCTrainValDataset(voc_root,
            class_names,
            split=val_split,
            transforms=val_transform)

    voc_val_dataloader = torch.utils.data.DataLoader(voc_val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=opt.batch_size,
        num_workers=0,
        drop_last=False)

    train_dataloader = voc_train_dataloader
    val_dataloader = voc_val_dataloader

    if TEST_DATASET_HAS_OPEN:
        test_list = "./datasets/test.txt"  # 还没有

        test_dataset = ListTestDataset(test_list, scale=opt.scale, max_size=max_size, norm=opt.norm_input)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    else:
        test_dataloader = None
