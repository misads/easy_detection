# encoding: utf-8
import torch
import ipdb
import cv2
import numpy as np
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
writer = create_summary_writer('logs/preview_dataloader')

class_names = opt.class_names
# class_names = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]



from utils.vis import visualize_boxes
for i, sample in enumerate(train_dataloader):
    # if i > 30:
    #     break
    utils.progress_bar(i, len(train_dataloader), 'Handling...')
    if opt.debug:
        ipdb.set_trace()

    image = sample['image'][0].detach().cpu().numpy().transpose([1,2,0])
    image = (image.copy()*255).astype(np.uint8)

    bboxes = sample['bboxes'][0].cpu().numpy()
    labels = sample['labels'][0].cpu().numpy().astype(np.int32)


    visualize_boxes(image=image, boxes=bboxes, labels=labels, probs=np.array(np.random.randint(85, 100,size=[len(bboxes)])/100), class_labels=class_names)

    write_image(writer, f'preview/{i}', 'image', image, 0, 'HWC')

writer.flush()
