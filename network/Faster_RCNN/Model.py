import pdb
import sys

import numpy as np
import torch
import cv2
import os

from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from mscv.summary import write_image

import misc_utils as utils
import ipdb

from torchvision.models.detection.faster_rcnn import FasterRCNN
from dataloader.coco import coco_90_to_80_classes

from .backbones import vgg16_backbone


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        kargs = {}
        if opt.scale:
            min_size = opt.scale
            max_size = int(min_size / 3 * 4)
            kargs = {'min_size': min_size,
                     'max_size': max_size
                    }

        # 定义backbone和Faster RCNN模型
        if opt.backbone is None or opt.backbone.lower() in ['res50', 'resnet', 'resnet50']:
            self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, **kargs)

            in_features = self.detector.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.num_classes + 1)

        elif opt.backbone.lower() in ['vgg16', 'vgg']:
            backbone = vgg16_backbone()
            self.detector = FasterRCNN(backbone, num_classes=opt.num_classes + 1, , **kargs)
        else:
            raise NotImplementedError, f'no such backbone: {opt.backbone }'


        print_network(self.detector)

        self.optimizer = get_optimizer(opt, self.detector)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, sample, *arg):
        """
        给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        labels = sample['labels']
        for label in labels:
            label += 1.  # effdet的label从1开始

        image, bboxes, labels = sample['image'], sample['bboxes'], sample['labels']
        
        if len(bboxes[0]) == 0:  # 没有bbox，不更新参数
            return {}

        image = image.to(opt.device)
        bboxes = [bbox.to(opt.device).float() for bbox in bboxes]
        labels = [label.to(opt.device).float() for label in labels]
        image = list(im for im in image)

        b = len(bboxes)

        target = [{'boxes': bboxes[i], 'labels': labels[i].long()} for i in range(b)]
        """
            target['boxes'] = boxes
            target['labels'] = labels
            # target['masks'] = None
            target['image_id'] = torch.tensor([index])
            target['area'] = area
            target['iscrowd'] = iscrowd
        """
        loss_dict = self.detector(image, target)

        loss = sum(l for l in loss_dict.values())

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward_test(self, image):  # test
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        conf_thresh = 0.5

        image = list(im for im in image)

        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        with torch.no_grad():
            outputs = self.detector(image)

        for b in range(len(outputs)):  #
            output = outputs[b]
            boxes = output['boxes']
            labels = output['labels']
            scores = output['scores']
            boxes = boxes[scores > conf_thresh]
            labels = labels[scores > conf_thresh]
            labels = labels.detach().cpu().numpy()
            # for i in range(len(labels)):
            #     labels[i] = coco_90_to_80_classes(labels[i])

            labels = labels - 1
            scores = scores[scores > conf_thresh]

            batch_bboxes.append(boxes.detach().cpu().numpy())
            batch_labels.append(labels)
            batch_scores.append(scores.detach().cpu().numpy())

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)
