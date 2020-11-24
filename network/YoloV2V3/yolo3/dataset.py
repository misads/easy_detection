#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .utils import read_truths_args, read_truths
from .image import *

def custom_collate(batch):
    data = torch.stack([item[0] for item in batch], 0)
    targets = torch.stack([item[1] for item in batch], 0)
    gts = torch.stack([item[2] for item in batch], 0)
    return data, targets, gts

class listDataset(Dataset):
    def __init__(self, root, gt_root, shape=None, shuffle=True, crop=False, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       with open(gt_root, 'r') as file:
           self.gts = file.readlines()

       # if shuffle:
       #     random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

       self.crop = crop
       self.jitter = jitter
       self.hue = hue
       self.saturation = saturation
       self.exposure = exposure

    def __len__(self):
        return self.nSamples

    def get_different_scale(self):
        if self.seen < 4000*self.batch_size:
            wh = 13*32                          # 416
        elif self.seen < 8000*self.batch_size:
            wh = (random.randint(0,3) + 13)*32  # 416, 480
        elif self.seen < 12000*self.batch_size:
            wh = (random.randint(0,5) + 12)*32  # 384, ..., 544
        elif self.seen < 16000*self.batch_size:
            wh = (random.randint(0,7) + 11)*32  # 352, ..., 576
        else: # self.seen < 20000*self.batch_size:
            wh = (random.randint(0,9) + 10)*32  # 320, ..., 608
        return (wh, wh)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        gtpath = self.gts[index].rstrip()

        if self.train:
            # if self.seen % (self.batch_size * 10) == 0: # in paper, every 10 batches, but we did every 64 images
            #     self.shape = self.get_different_scale()
            img, label = load_data_detection(imgpath, self.shape, self.crop, self.jitter, self.hue, self.saturation, self.exposure)
            gt = Image.open(gtpath).convert('RGB')

            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            # if self.shape:
            #     img, org_w, org_h = letterbox_image(img, self.shape[0], self.shape[1]), img.width, img.height
            org_w = org_h = 256


            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            gt = self.transform(gt)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        if self.train:
            return (img, label, gt)
        else:
            return (img, label, org_w, org_h)
