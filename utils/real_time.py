import torch
torch.multiprocessing.set_sharing_strategy('file_system')


from options import opt
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im
from network import get_model

import misc_utils as utils
import ipdb

import cv2
import numpy as np
import time
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from options import opt
from utils.vis import visualize_boxes

import misc_utils as utils

class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

config.DATA.CLASS_NAMES = class_names
config.DATA.NUM_CLASSESS = len(class_names)

opt.width = 600
opt.height = 600


# FRCNN
divisor = 32
val_transform = A.Compose(
    [
        # A.SmallestMaxSize(600, p=1.0),  # resize到短边600
        # A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0
)

Model = get_model(config.MODEL.NAME)
model = Model(opt)
model = model.to(device=opt.device)


if opt.load:
    which_epoch = model.load(opt.load)
else:
    which_epoch = 0

model.eval()

# img = cv2.imread('a.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
# img /= 255.0  # 转成0~1之间
#
# img_input = val_transform(image=img)['image']
# img_input = img_input.unsqueeze(0).to(opt.device)
#
# batch_bboxes, batch_labels, batch_scores = model(img_input)
#
# img = tensor2im(img_input).copy()
# # for x1, y1, x2, y2 in gt_bbox[0]:
# #     cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)  # 绿色的是gt
#
# num = len(batch_scores[0])
# visualize_boxes(image=img, boxes=batch_bboxes[0],
#                 labels=batch_labels[0].astype(np.int32), probs=batch_scores[0], class_labels=config.DATA.CLASS_NAMES)
#
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow('image', img)
# cv2.waitKey(0)

cap = cv2.VideoCapture(0)

print(cap.isOpened())

cap.set(3, 1024)
cap.set(4, 768)

frames = 0
fps = 0
t = time.time()

while True:
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0  # 转成0~1之间

    img_input = val_transform(image=img)['image']
    img_input = img_input.unsqueeze(0).to(opt.device)

    batch_bboxes, batch_labels, batch_scores = model(img_input)

    img = tensor2im(img_input).copy()
    # for x1, y1, x2, y2 in gt_bbox[0]:
    #     cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)  # 绿色的是gt

    num = len(batch_scores[0])
    visualize_boxes(image=img, boxes=batch_bboxes[0],
                    labels=batch_labels[0].astype(np.int32), probs=batch_scores[0], class_labels=config.DATA.CLASS_NAMES)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img=np.rot90(img)
    frames = frames + 1
    nt = time.time()
    if nt - t > 1.0:
        fps = frames
        frames = 0
        t = nt

        # Draw fps

    cv2.putText(img, 'fps:' + str(fps), (35, 0 + 30), 0, 1, (0, 255, 0), 2)

    cv2.imshow('easy_detection', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

