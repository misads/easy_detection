# encoding=utf-8
from dataloader.image_folder import FolderTrainValDataset, FolderTestDataset
from dataloader.image_list import ListTrainValDataset, ListTestDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader import yolo_dataset
from dataloader import paired_dataset

from options import opt
from mscv.image import tensor2im
import torch
import pdb

###################

TEST_DATASET_HAS_OPEN = False  # 有没有开放测试集

###################

train_list = "./datasets/apollo/merge_train.txt"
val_list = "./datasets/apollo/apollo_val.txt"

max_size = 64 if opt.debug else None

train_dataset = ListTrainValDataset(train_list, scale=opt.scale, crop=opt.crop, aug=opt.aug, max_size=max_size, norm=opt.norm_input)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

val_dataset = ListTrainValDataset(val_list, scale=opt.scale, aug=False, max_size=max_size, norm=opt.norm_input)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)


src_list = "./datasets/yolo/apollo_clear_train.txt"
src_val_list = "./datasets/yolo/apollo_clear_val.txt"
"""
source domain 是clear的
"""

num_workers = 3

src_data_loader = torch.utils.data.DataLoader(
            yolo_dataset.listDataset(src_list, 
                                shape=(416, 416),
                                shuffle=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]), 
                                train=True,
                                seen=0,
                                batch_size=opt.batch_size,
                                num_workers=num_workers),
            collate_fn=yolo_dataset.custom_collate, 
            batch_size=opt.batch_size, shuffle=False, num_workers=num_workers)


src_val_loader = torch.utils.data.DataLoader(
    yolo_dataset.listDataset(src_val_list,
                        shape=(416, 416),
                        shuffle=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]), train=False),
    batch_size=1, shuffle=False, num_workers=3)


tgt_list = "./datasets/yolo/apollo_train.txt"
tgt_val_list = "./datasets/yolo/apollo_val.txt"
"""
target domain 是hazy的
"""

tgt_data_loader = torch.utils.data.DataLoader(
            yolo_dataset.listDataset(tgt_list, 
                                shape=(416, 416),
                                shuffle=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    ]), 
                                train=True,
                                seen=0,
                                batch_size=opt.batch_size,
                                num_workers=num_workers),
            collate_fn=yolo_dataset.custom_collate, 
            batch_size=opt.batch_size, shuffle=False, num_workers=num_workers)


tgt_val_loader = torch.utils.data.DataLoader(
    yolo_dataset.listDataset(tgt_val_list,
                        shape=(416, 416),
                        shuffle=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]), train=False),
    batch_size=1, shuffle=False, num_workers=3)


paired_list = './datasets/apollo/apollo_val.txt'  # 这里要填一下

paired_loader = torch.utils.data.DataLoader(    
    paired_dataset.pairedDataset(paired_list, shape=(416, 416),  # 这里不是listDataset
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])), 
    batch_size=opt.batch_size, shuffle=False, num_workers=3)

# ([b, 3, 416, 416], [b, 250])
"""
50×5 最多50个bbox
"""


sots_list = './datasets/ITS_val.txt'
sots_dataset = ListTrainValDataset(sots_list, scale=opt.scale, aug=False, max_size=max_size, norm=opt.norm_input)
sots_dataloader = DataLoader(sots_dataset, batch_size=1, shuffle=False, num_workers=1)


sots_outdoor_list = './datasets/SOTS_OUTDOOR.txt'
sots_outdoor_dataset = ListTrainValDataset(sots_outdoor_list, scale=opt.scale, aug=False, max_size=max_size, norm=opt.norm_input)
sots_outdoor_dataloader = DataLoader(sots_outdoor_dataset, batch_size=1, shuffle=False, num_workers=1)


hsts_list = './datasets/HSTS.txt'
hsts_dataset = ListTrainValDataset(hsts_list, scale=opt.scale, aug=False, max_size=max_size, norm=opt.norm_input)
hsts_dataloader = DataLoader(hsts_dataset, batch_size=1, shuffle=False, num_workers=1)


real_list = "./datasets/REAL.txt" 
real_dataset = ListTestDataset(real_list, scale=opt.scale, max_size=max_size, norm=opt.norm_input)
real_dataloader = DataLoader(real_dataset, batch_size=1, shuffle=False, num_workers=1)


if TEST_DATASET_HAS_OPEN:
    test_list = "./datasets/test.txt"  # 还没有

    test_dataset = ListTestDataset(test_list, scale=opt.scale, max_size=max_size, norm=opt.norm_input)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

else:
    test_dataloader = None
