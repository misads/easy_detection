# encoding: utf-8
import torch
import ipdb
import cv2

from options import opt
# from dataloader import paired_dataset
from mscv.summary import create_summary_writer, write_image
from mscv.image import tensor2im

from dataloader.dataloaders import train_dataloader, val_dataloader
from dataloader import voc
import misc_utils as utils

"""
source domain 是clear的
"""
writer = create_summary_writer('logs/preview_dataset')

cityscapes_classes = ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']

colors = [(0.,1.,0.), (0.,0.,1.), (1.,0.,0.), (0.,1.,1.), (1.,0.,1.), (1.,1.,0.), (1.,1.,1.), (0.,0.,0.), (.5,0.,0.), 
(0.,.5,0.), (0.,0.,.5), (0.,.5,.5), (.5,0.,.5), (.5,.5,0.), (.5,.5,.5), (.5,1.,0.), (.5,0.,1.), (1.,.5,0.), 
(0.,.5,1.), (1.,0.,.5), (0.,1.,.5), (.5,.5,1.), (.5,1.,.5), (1.,.5,.5), (1.,1.,.5), (1.,.5,1.), (.5,1.,1.)]


for i, sample in enumerate(val_dataloader):
    # if i > 30:
    #     break
    utils.progress_bar(i, len(val_dataloader), 'Handling...')
    continue

    if opt.debug:
        ipdb.set_trace()

    image = sample['image'][0].detach().cpu().numpy().transpose([1,2,0])
    image = image.copy()
    bboxes = sample['bboxes'][0]
    labels = sample['labels'][0]
    

    for j, (x1, y1, x2, y2) in enumerate(bboxes):
        x1 = int(round(x1.item()))
        y1 = int(round(y1.item()))
        x2 = int(round(x2.item()))
        y2 = int(round(y2.item()))
        label = int(labels[j].item())
        label_name = cityscapes_classes[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 1., 0), 2)
        cv2.putText(image, label_name, (x1, y1-3), 0, 1, colors[label], 2)

    write_image(writer, f'preview/{i}', 'image', image, 0, 'HWC')

writer.flush()
