# encoding=utf-8
from torch.utils.data import DataLoader, DistributedSampler
from dataloader.list_dataset import get_all_datasets

from options import opt, config
from options.helper import is_distributed

import torch

def collate_fn(batch):
    target = {}
    b = len(batch)
    target['image'] = [sample['image'] for sample in batch]   # torch.stack([sample['image'] for sample in batch])
    target['bboxes'] = [sample['bboxes'] for sample in batch]
    target['labels'] = [sample['labels'] for sample in batch]
    target['path'] = [sample['path'] for sample in batch]
    target['yolo_boxes'] = torch.stack([sample['yolo_boxes'] for sample in batch])
    target['yolo4_boxes'] = torch.stack([sample['yolo4_boxes'] for sample in batch])
    target['yolo5_boxes'] = torch.cat(  # [b*50, 6] batch中第几张图片, label, c_x, c_y, w, h
                                [torch.cat([torch.ones([batch[i]['yolo5_boxes'].shape[0], 1]) * i,
                                batch[i]['yolo5_boxes']], 1) for i in range(b)], 0)
    return target

# Datasets
train_dataset, val_dataset = get_all_datasets()

# Dataloaders
if train_dataset is None:
    train_dataloader = None
else:
    sampler = DistributedSampler(train_dataset) if is_distributed() else None
    shuffle = not is_distributed() 

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=config.OPTIMIZE.BATCH_SIZE,
        num_workers=config.MISC.NUM_WORKERS,
        drop_last=True,
        sampler=sampler)


if val_dataset is None:
    val_dataloader = None
else:
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config.OPTIMIZE.BATCH_SIZE,
        num_workers=config.MISC.NUM_WORKERS // 2,
        )
