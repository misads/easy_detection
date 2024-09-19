import pdb
import sys
import numpy as np
import torch
import cv2
import os

from torch import nn
import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .retinanet import resnet50 as Retina_50
from options import opt

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from mscv.summary import write_image


class Model(BaseModel):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, **kwargs)
        self.opt = opt
        self._detector = Retina_50(config.DATA.NUM_CLASSESS,pretrained=True)
        self.init_common()

    def update(self, sample, *arg):
        """
        给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        images = sample['image'].to(opt.device)


        annotations = [torch.cat([box, label.unsqueeze(1)], dim=1).to(opt.device) for box, label in zip(sample['bboxes'],sample['labels'])]
        inputs = images, annotations

        loss = sum(self.detector(inputs))

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward_test(self, sample):  # test
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        image = sample['image']

        conf_thresh = 0.  # 0.5 for vis result

        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        for i in range(image.shape[0]):
            single_image = image[i: i+1]

            scores, labels, bboxes = self.detector(single_image)  # RetinaNet只支持单张检测

            conf = scores > conf_thresh
            bboxes = bboxes[conf].detach().cpu().numpy()
            labels = labels[conf].detach().cpu().numpy()
            scores = scores[conf].detach().cpu().numpy()

            batch_bboxes.append(bboxes)
            batch_labels.append(labels)
            batch_scores.append(scores)

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError
