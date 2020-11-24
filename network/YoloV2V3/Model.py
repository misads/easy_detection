import pdb
import sys
sys.path.insert(0, "./timm-efficientdet-pytorch")
sys.path.insert(0, "./omegaconf")

import numpy as np
import torch
import os

from torch import nn
import gc
from .yolo3.darknet import Darknet
from .yolo3.utils import get_all_boxes, bbox_iou, nms, read_data_cfg, load_class_names
from .yolo3.image import correct_yolo_boxes
# from .yolo3.utils import *
from .yolo3.eval_map import eval_detection_voc


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


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        
        # 根据YoloV2和YoloV3使用不同的配置文件
        if opt.model == 'Yolo2':
            cfgfile = 'configs/yolo2-voc.cfg'
        elif opt.model == 'Yolo3':
            cfgfile = 'configs/yolo3-coco.cfg'

        # 初始化detector
        self.detector = Darknet(cfgfile, device=opt.device).to(opt.device)
        print_network(self.detector)

        # 在--load之前加载weights文件(可选)
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
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        loss_layers = self.detector.loss_layers
        org_loss = []

        image = sample['image'].to(opt.device)  # target domain
        target = sample['yolo_boxes'].to(opt.device)

        detection_output = self.detector(image)  # 在src domain上训练检测

        for i, l in enumerate(loss_layers):
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
            shape = (opt.width, opt.height)

        num_classes = self.detector.num_classes

        output = self.detector(image)
        all_boxes = get_all_boxes(output, shape, conf_thresh, num_classes,
                                  device=opt.device, only_objectness=0,
                                  validation=True)

        for b in range(len(all_boxes)):
            boxes = all_boxes[b]
            width = opt.width
            height = opt.height
            correct_yolo_boxes(boxes, width, height, width, height)

            boxes = nms(boxes, nms_thresh)
            img_boxes = []
            img_labels = []
            img_scores = []
            for box in boxes:
                box[0] -= box[2] / 2
                box[1] -= box[3] / 2
                box[2] += box[0]
                box[3] += box[1]

                box[0] *= width
                box[2] *= width
                box[1] *= height
                box[3] *= height

                for i in range(5, len(box), 2):
                    img_boxes.append(box[:4])
                    img_labels.append(box[i+1])
                    score = box[4] * box[i]
                    img_scores.append(score)

            batch_bboxes.append(np.array(img_boxes))
            batch_labels.append(np.array(img_labels).astype(np.int32))
            batch_scores.append(np.array(img_scores))

                
            # boxes = np.array([box[:7] for box in boxes])

            # """cxcywh转xyxy"""
            # boxes[:, 0] -= boxes[:, 2] / 2
            # boxes[:, 1] -= boxes[:, 3] / 2
            # boxes[:, 2] += boxes[:, 0]
            # boxes[:, 3] += boxes[:, 1]

            # boxes[:, 0] *= width
            # boxes[:, 2] *= width
            # boxes[:, 1] *= height
            # boxes[:, 3] *= height

            # score = boxes[:, 4] * boxes[:, 5]

            # # conf_indics = score > 0.5
            # # score = score[conf_indics]
            # # boxes = boxes[conf_indics]

            # batch_bboxes.append(boxes[:, :4])
            # batch_labels.append(boxes[:, 6].astype(np.int32))
            # batch_scores.append(score)

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)



