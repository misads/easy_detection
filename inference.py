# encoding:utf-8
import os
import cv2
import torch
import ipdb
import numpy as np
from misc_utils import get_file_name # pip install utils-misc
import albumentations as A

from network import get_model
from options import opt, config
from utils import visualize_boxes
from torchvision.ops import nms
from albumentations.pytorch.transforms import ToTensorV2

class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

LOAD_CHECKPOINT = 'pretrained/0_voc_FasterRCNN.pt'
INFERENCE_LIST = 'datasets/voc/ImageSets/Main/test.txt'
IMAGE_FOLDER = 'datasets/voc/JPEGImages'
KEEP_THRESH = 0.5
SAVE_PATH = 'results/inference'


class Option:
    model = 'Faster_RCNN'
    transform = 'frcnn'
    backbone = None
    scale = None
    num_classes = len(class_names)
    device = 'cuda:0'
    FORMAT = 'jpg'

    # ===========没什么用的===========
    optimizer = 'sgd'
    scheduler = 'cos'
    lr = 0.1
    epochs = 12
    checkpoint_dir = 'checkpoints'
    tag = 'inference'


opt = Option()



Model = get_model(config.MODEL.NAME)
model = Model(opt)
model = model.to(device=opt.device)
model.load(LOAD_CHECKPOINT)
model.eval()

test_transform = A.Compose([ToTensorV2(p=1.0)], p=1.0)


def inference(image_path, keep_thresh=0.5, savepath=None):
    """
    Args: 
        image_path: image path
        keep_thresh: 置信度大于多少的保留
        savepath: 预览图保存路径，为None不保存
    Returns:
        a tuple (bboxes, labels, scores)
        bboxes:
            [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        labels:
            [label1, label2, ...]
        scores:
            [score1, score2, ...]  # 降序排列
    """
    assert os.path.isfile(image_path), f'{image_path} not exists.'
    filaname = get_file_name(image_path)
    
    image_org = cv2.imread(image_path)
    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image_org / 255.0

    sample = test_transform(**{
        'image': image,
    })
    image = sample['image']
    image = image.unsqueeze(0).to(opt.device)
    batch_bboxes, batch_labels, batch_scores = model.forward_test(image)
    bboxes = batch_bboxes[0]
    labels = batch_labels[0]
    scores = batch_scores[0]
    keep = scores > keep_thresh
    bboxes = bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    R = lambda x: int(round(x, 0))  # 四舍五入
    visualize_boxes(image=image_org, boxes=bboxes, labels=labels, probs=scores, class_labels=class_names)

    bboxes_arr = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = bboxes[i]
        x1, y1, x2, y2 = R(x1), R(y1), R(x2), R(y2) 
        bboxes_arr.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes_arr)

    if savepath:
        print(f'save image to {savepath}/{filaname}.png')
        image = cv2.cvtColor(np.asarray(image_org), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{savepath}/{filaname}.png', image)

    return bboxes, labels, scores

if __name__ == '__main__':
    val_txt = INFERENCE_LIST
    image_folder = IMAGE_FOLDER

    with open(val_txt, 'r') as f:
        lines = f.readlines()

    os.makedirs(SAVE_PATH, exist_ok=True)

    for i, line in enumerate(lines):
        print(f'{i}/{len(lines)}')
        line = line.rstrip('\n')
        image_path = os.path.join(image_folder, f'{line}.{opt.FORMAT}')
        bboxes, labels, scores = inference(image_path, keep_thresh=KEEP_THRESH, savepath=SAVE_PATH)
        

