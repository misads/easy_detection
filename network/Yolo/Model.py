import pdb
import sys
sys.path.insert(0, "./timm-efficientdet-pytorch")
sys.path.insert(0, "./omegaconf")

import numpy as np
import torch
import os

from torch import nn
import gc
from yolo3.darknet import Darknet
from yolo3.utils import get_all_boxes, bbox_iou, nms, read_data_cfg, load_class_names
from yolo3.image import correct_yolo_boxes
from yolo3.utils import *
from yolo3.image import correct_yolo_boxes
from yolo3.eval_map import eval_detection_voc

from .eval_yolo import eval_yolo

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from mscv.summary import write_loss
# from mscv.cnn import normal_init
from loss import get_default_loss

import misc_utils as utils
import ipdb

conf_thresh = 0.005
nms_thresh = 0.45
iou_thresh = 0.5


def truths_length(truths):
    for i in range(50):
        if truths[i][1] == 0:
            return i
    return 50


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        
        if opt.model == 'Yolo2':
            cfgfile = 'yolo-voc.cfg'
        elif opt.model == 'Yolo3':
            cfgfile = 'yolo_v3.cfg'

        self.detector = Darknet(cfgfile, device=opt.device).to(opt.device)
        #####################
        #    Init weights
        #####################
        # normal_init(self.detector)

        print_network(self.detector)

        if opt.weights:
            utils.color_print('Load Yolo weights from %s.' % opt.weights, 3)
            self.detector.load_weights(opt.weights)

        self.optimizer = get_optimizer(opt, self.detector)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, sample, *arg):
        """
        Args:
            sample: {'input': input_image [b, 3, height, width],
                   'bboxes': bboxes [b, None, 4],
                   'labels': labels [b, None],
                   'path': paths}
        """
        loss_layers = self.detector.loss_layers
        org_loss = []

        image = sample['image'].to(opt.device)  # target domain
        target = sample['yolo_boxes'].to(opt.device)

        detection_output = self.detector(image)  # 在src domain上训练检测

        for i, l in enumerate(loss_layers):
            # l.seen = l.seen + data.data.size(0)
            ol=l(detection_output[i]['x'], target)
            org_loss.append(ol)

        loss = sum(org_loss)

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.detector.parameters(), 10000)
        self.optimizer.step()

        org_loss.clear()

        gc.collect()
        return {}

    def forward(self, image):
        """
        Args:
            image: [b, 3, h, w] Tensor
        """
        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        if self.detector.net_name() == 'region':  # region_layer
            shape = (0, 0)
        else:
            shape = (512, 512)

        num_classes = self.detector.num_classes

        output = self.detector(image)
        all_boxes = get_all_boxes(output, shape, conf_thresh, num_classes,
                                  device=opt.device, only_objectness=0,
                                  validation=True)

        for b in range(len(all_boxes)):
            boxes = all_boxes[b]
            width = 1024
            height = 512
            correct_yolo_boxes(boxes, width, height, width, height)
            boxes = nms(boxes, nms_thresh)

            boxes = np.array([box[:7] for box in boxes])

            """cxcywh转xyxy"""
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height

            score = boxes[:, 4] * boxes[:, 5]


            conf_indics = score > 0.5
            score = score[conf_indics]
            boxes = boxes[conf_indics]

            batch_bboxes.append(boxes[:, :4])
            batch_labels.append(boxes[:, 6].astype(np.int32))
            batch_scores.append(score)

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)



