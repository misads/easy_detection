import pdb
import sys
import numpy as np
import torch
import cv2
import os

from .ssd import SSDDetector
from .transform.target_transform import SSDTargetTransform
from .anchors.prior_box import PriorBox

from options import opt

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from mscv.summary import write_image
# from mscv.cnn import normal_init


class Model(BaseModel):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, **kwargs)
        self.config = config
        self.detector = SSDDetector(config).to(device=opt.device)
        #####################
        #    Init weights
        #####################
        # normal_init(self.detector)
        self.init_common()

        CENTER_VARIANCE = 0.1
        SIZE_VARIANCE = 0.2
        THRESHOLD = 0.5

        self.target_transform = SSDTargetTransform(PriorBox(config)(),
                                   CENTER_VARIANCE,
                                   SIZE_VARIANCE,
                                   THRESHOLD)

    def update(self, sample, *arg):
        """
        给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        loss_dict = self.forward_train(sample)

        loss = sum(loss for loss in loss_dict.values())
        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward_train(self, sample):
        labels = sample['labels']
        for label in labels:
            label += 1.  # effdet的label从1开始

        image, bboxes, labels = sample['image'], sample['bboxes'], sample['labels']

        image = image * 255 
        sub = torch.Tensor([[123, 117, 104]]).view([1, 3, 1, 1])
        image -= sub

        for box in bboxes:
            box[:, 0::2] /= 512  # width
            box[:, 1::2] /= 512  # height

        for b in range(len(labels)):
            bboxes[b], labels[b] = self.target_transform(bboxes[b], labels[b])

        image = image.to(opt.device)
        bboxes = torch.stack(bboxes).to(opt.device)
        labels = torch.stack(labels).long().to(opt.device)

        targets = {'boxes': bboxes, 'labels': labels}

        loss_dict = self.detector(image, targets=targets)

        return loss_dict

    def forward_test(self, sample):
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        conf_thresh = 0.000
        image = sample['image']

        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        """
        图像预处理
        """
        image = image * 255 
        sub = torch.Tensor([[123, 117, 104]]).cuda().view([1, 3, 1, 1])
        image -= sub

        outputs = self.detector(image)

        """
        nms阈值设置在 .box_head/inference.py
        """
        for b in range(len(outputs)):  #
            output = outputs[b]
            boxes = output['boxes']
            labels = output['labels']
            scores = output['scores']
            labels = labels - 1
            batch_bboxes.append(boxes.detach().cpu().numpy())
            batch_labels.append(labels.detach().cpu().numpy())
            batch_scores.append(scores.detach().cpu().numpy())

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError



