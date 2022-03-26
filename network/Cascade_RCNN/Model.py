import pdb
import sys

import numpy as np
import torch
import cv2
import os

from torch import nn
import torch

from options import opt
from options.helper import is_distributed, is_first_gpu

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from mscv.summary import write_image

import misc_utils as utils
import ipdb

from .frcnn.cascade_rcnn import CascadeRCNN, FastRCNNPredictor
from .frcnn.rpn import AnchorGenerator
from .frcnn import cascadercnn_resnet50_fpn

from .backbones import vgg16_backbone, res101_backbone


class Model(BaseModel):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, kwargs)
        self.config = config

        kargs = {}
        if 'SCALE' in config.DATA:
            scale = config.DATA.SCALE
            if isinstance(scale, int):
                min_size = scale
                max_size = int(min_size / 3 * 5)
            else:
                min_size, max_size = config.DATA.SCALE

            kargs = {'min_size': min_size,
                     'max_size': max_size,
                    }
        
        kargs.update({'box_nms_thresh': config.TEST.NMS_THRESH})

        # 多卡使用 SyncBN
        if is_distributed():
            kargs.update({'norm_layer': torch.nn.SyncBatchNorm})

        # 定义backbone和Faster RCNN模型
        if config.MODEL.BACKBONE is None or config.MODEL.BACKBONE.lower() in ['res50', 'resnet50']:
            # 默认是带fpn的resnet50
            self.detector = cascadercnn_resnet50_fpn(pretrained=False, **kargs)

            in_features = self.detector.roi_heads[0].box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            self.detector.roi_heads[0].box_predictor = FastRCNNPredictor(in_features, config.DATA.NUM_CLASSESS + 1)
            self.detector.roi_heads[1].box_predictor = FastRCNNPredictor(in_features, config.DATA.NUM_CLASSESS + 1)
            self.detector.roi_heads[2].box_predictor = FastRCNNPredictor(in_features, config.DATA.NUM_CLASSESS + 1)

        elif config.MODEL.BACKBONE.lower() in ['vgg16', 'vgg']:
            backbone = vgg16_backbone()
            self.detector = CascadeRCNN(backbone, num_classes=config.DATA.NUM_CLASSESS + 1, **kargs)

        elif config.MODEL.BACKBONE.lower() in ['res101', 'resnet101']:
            # 不带FPN的resnet101
            backbone = res101_backbone()
            self.detector = CascadeRCNN(backbone, num_classes=config.DATA.NUM_CLASSESS + 1, **kargs)

        elif config.MODEL.BACKBONE.lower() in ['res', 'resnet']:
            raise RuntimeError(f'backbone "{config.MODEL.BACKBONE}" is ambiguous, please specify layers.')

        else:
            raise NotImplementedError(f'no such backbone: {config.MODEL.BACKBONE}')

        if opt.debug:
            print_network(self.detector)

        self.to(opt.device)
        # 多GPU支持
        if is_distributed():
            self.detector = torch.nn.parallel.DistributedDataParallel(self.detector, find_unused_parameters=False,
                    device_ids=[opt.local_rank], output_device=opt.local_rank)
            # self.detector = torch.nn.parallel.DistributedDataParallel(self.detector, device_ids=[opt.local_rank], output_device=opt.local_rank)

        self.optimizer = get_optimizer(config, self.detector)
        self.scheduler = get_scheduler(config, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join('checkpoints', opt.tag)

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
 
        for b in range(len(image)):
            if len(bboxes[b]) == 0:  # 没有bbox，不更新参数
                return {}

        #image = image.to(opt.device)
        bboxes = [bbox.to(opt.device).float() for bbox in bboxes]
        labels = [label.to(opt.device).float() for label in labels]
        image = list(im.to(opt.device) for im in image)

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
        # try:
        loss_dict = self.detector(image, target)
        # except:
        #     return {}
            # import ipdb
            # ipdb.set_trace()

        loss = sum(l for l in loss_dict.values())

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}


    def forward_test(self, image):  # test
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        conf_thresh = 0.05

        image = list(im.to(opt.device) for im in image)

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

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        self.detector.load_state_dict(state['detector'])
        # return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)
