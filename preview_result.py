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
from mscv.summary import create_summary_writer, write_image

def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    

    config.num_classes = 1
    config.image_size=512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')

    net.load_state_dict(checkpoint['detector'])
    # net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchEval(net, config)
    net.eval()
    return net

net = load_net('./checkpoints/transform/40_Effdet.pt')

# net = load_net('./checkpoints/best-checkpoint-065epoch.bin')
net.to('cuda:1')

from dataloader import voc

val_transform =A.Compose(
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

voc_dataset = voc.VOCTrainValDataset('/home/raid/public/datasets/wheat_detection', 
        ['wheat'],
        split='val.txt',
        transforms=val_transform)

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
            labels = det[i].detach().cpu().numpy()[:,5]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
                'labels': labels[indexes]
            })
    return [predictions]

# import ipdb; ipdb.set_trace()

writer = create_summary_writer('results/preview_result')

for i, sample in enumerate(val_dataloader):
    if i > 20:
        break

    images = sample['input'].to('cuda:1')
    predictions = make_predictions(images)
    img = images[0].detach().cpu().numpy().transpose([1,2,0])
    img = img.copy()
    bboxes = predictions[0][0]['boxes']

    for x1, y1, x2, y2 in bboxes:
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0., 1., 0.), 2)

    write_image(writer, f'result/{i}', 'predicted', img, 0, 'HWC')
        