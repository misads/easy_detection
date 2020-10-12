# encoding=utf-8
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader.custom import get_dataset
from dataloader.voc import VOCTrainValDataset
from dataloader.coco import CocoDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from options import opt
from mscv.image import tensor2im
import torch
import pdb


###################

TEST_DATASET_HAS_OPEN = False  # 有没有开放测试集
DATA_FOTMAT = 'COCO'  # 数据集格式

###################



if DATA_FOTMAT == 'VOC':

    opt.dataset = 'wheat'

    dataset = get_dataset(opt.dataset)
    d = dataset()

    variable_names = ['voc_root', 'train_split', 'val_split', 'class_names', 'img_format', 
                      'width', 'height', 'train_transform', 'val_transform']

    for v in variable_names:
        # 等价于 exec(f'{v}=d.{v}')
        locals()[v] = getattr(d, v)  # 把类的成员变量赋值给当前的局部变量

    opt.class_names = class_names
    opt.num_classes = len(class_names)

    opt.width = width
    opt.height = height

    train_dataset = VOCTrainValDataset(voc_root, 
            class_names,
            split=train_split,
            format=img_format,
            transforms=train_transform)

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

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        drop_last=True)

    val_dataset = VOCTrainValDataset(voc_root,
            class_names,
            split=val_split,
            format=img_format,
            transforms=val_transform)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=opt.batch_size,
        num_workers=opt.workers // 2,
        drop_last=False)

    if TEST_DATASET_HAS_OPEN:
        test_list = "./datasets/test.txt"  # 还没有

        test_dataset = ListTestDataset(test_list, scale=opt.scale, max_size=max_size, norm=opt.norm_input)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    else:
        test_dataloader = None

elif DATA_FOTMAT == 'COCO':

    def collate_fn(batch):
        target = {}
        b = len(batch)
        target['image'] = torch.stack([sample['image'] for sample in batch])
        target['bboxes'] = [sample['bboxes'] for sample in batch]
        target['labels'] = [sample['labels'] for sample in batch]

        return target
    transform = A.Compose(
        [
            A.Resize(height=512, width=512, p=1),
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
    train_dataset = CocoDataset(transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   shuffle=True,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.workers,
                                                   collate_fn=collate_fn,
                                                   drop_last=True)

    val_dataloader = None

