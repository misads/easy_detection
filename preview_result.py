import sys
sys.path.insert(0, "./timm-efficientdet-pytorch")
sys.path.insert(0, "./omegaconf")


import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import gc
from matplotlib import pyplot as plt
from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet.efficientdet import HeadNet


def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    

    config.num_classes = 1
    config.image_size=512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')

    # net.load_state_dict(checkpoint['detector'])
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval()
    return net

# net = load_net('./checkpoints/effdet2/20_Effdet.pt')

net = load_net('./checkpoints/best-checkpoint-065epoch.bin')
net.to('cuda:1')

from dataloader import voc

voc_dataset = voc.VOCTrainValDataset('/home/raid/public/datasets/wheat_detection', 
        ['wheat'],
        split='val.txt',
        scale=512)

def collate_fn(batch):
    target = {}
    target['input'] = torch.stack([sample['input'] for sample in batch])
    target['bboxes'] = [sample['bboxes'] for sample in batch]
    target['labels'] = [sample['labels'] for sample in batch]
    target['path'] = [sample['path'] for sample in batch]
    return target

val_dataloader = torch.utils.data.DataLoader(voc_dataset,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=4,
    num_workers=3,
    drop_last=False)
    

def make_predictions(images, score_threshold=0.22):
    # images = torch.stack(images).cuda('cuda:1').float()
    predictions = []

    with torch.no_grad():
        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda('cuda:1'))
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

import ipdb; ipdb.set_trace()

for i, sample in enumerate(val_dataloader):
    images = sample['input'].to('cuda:1')
    predictions = make_predictions(images)
        