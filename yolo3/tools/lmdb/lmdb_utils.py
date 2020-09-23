#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
import cv2
import lmdb

class lmdbDataset(Dataset):

    def __init__(self, lmdb_root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0):
        self.env = lmdb.open(lmdb_root,
                 max_readers=1,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 meminit=False)
        self.txn = self.env.begin(write=False) 
        self.nSamples = int(self.txn.get('num-samples'))
        self.indices = range(self.nSamples) 
        if shuffle:
            random.shuffle(self.indices)
 
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        #if self.train:
        #    print('init seen to %d' % (self.seen))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgkey = 'image-%09d' % (self.indices[index]+1)
        labkey = 'label-%09d' % (self.indices[index]+1)
        label = torch.zeros(50*5)

        imageBin = self.txn.get(imgkey)
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        if self.train and index % 64 == 0:
            if self.seen < 4000*64*4:
               width = (random.randint(0,2)*2 + 13)*32
               self.shape = (width, width)
            elif self.seen < 8000*64*4:
               width = (random.randint(0,4)*2 + 9)*32
               self.shape = (width, width)
            elif self.seen < 12000*64*4:
               width = (random.randint(0,6)*2 + 5)*32
               self.shape = (width, width)
            elif self.seen < 12000*64*4:
               width = (random.randint(0,12) + 5)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64*4:
               width = (random.randint(0,16) + 3)*32
               self.shape = (width, width)

        if self.shape:
            img = cv2.resize(img, self.shape, interpolation = cv2.INTER_CUBIC)

        tid = 0
        truths = self.txn.get(labkey).rstrip().split('\n')
        for truth in truths:
            truth = truth.split()
            tmp = [float(t) for t in truth]
            if tmp[3] > 8.0/img.shape[0]:
                label[tid*5+0] = tmp[0]
                label[tid*5+1] = tmp[1]
                label[tid*5+2] = tmp[2]
                label[tid*5+3] = tmp[3]
                label[tid*5+4] = tmp[4]
                tid = tid + 1

        width = img.shape[0]
        height = img.shape[1]
        img = torch.from_numpy(img)
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + 4
        return (img, label)

def lmdb_nsamples(db):
    env = lmdb.open(db,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'))
    return nSamples

