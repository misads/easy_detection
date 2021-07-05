import pdb
import sys


import numpy as np
import torch
import cv2
import os

import ipdb
from torch import nn
import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .Yolov4 import Yolov4 as yolov4
from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from torchvision.ops import nms
from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from mscv.summary import write_image
from network.YoloV4.loss import Yolo_loss
from network.YoloV4 import config as cfg
from network.YoloV4 import tools

import misc_utils as utils


class Model(BaseModel):
    def __init__(self, opt, logger=None):
        super(Model, self).__init__(config, kwargs)
        self.opt = opt
        # cfgfile = 'yolo-voc.cfg'
        # self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        #
        # # replace the pre-trained head with a new one
        # self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.DATA.NUM_CLASSESS + 1)
        self.detector = yolov4(inference=True, n_classes=config.DATA.NUM_CLASSESS)
        
        # """
        # 预训练模型
        # """
        # pretrained_dict = torch.load('pretrained/yolov4.pth')
        # self.detector.load_state_dict(pretrained_dict)


        self.yolov4loss = Yolo_loss(device=opt.device,batch=opt.batch_size)
        #####################
        #    Init weights
        #####################
        # normal_init(self.detector)

        if opt.debug:
            print_network(self.detector)

        self.optimizer = get_optimizer(opt, self.detector)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join('checkpoints', opt.tag)

    def update(self, sample, *arg):
        """
        给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
        Args:
            sample: {'input': input_image [b, 3, height, width],
                   'bboxes': bboxes [b, None, 4],
                   'labels': labels [b, None],
                   'path': paths}
        """
        images = sample['image'].to(opt.device)
        bboxes = sample['yolo4_boxes'].to(opt.device)
        # bboxes = [torch.cat([box, label.unsqueeze(1)], dim=1).to(opt.device) for box, label in zip(sample['bboxes'],sample['labels'])]
        out = self.detector(images)
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = self.yolov4loss(out, bboxes)


        self.avg_meters.update({'loss': loss.item()})
        # 'loss_xy':loss_xy.item(), 'loss_wh':loss_wh.item(), 'loss_obj':loss_obj.item(),
        # 'loss_cls':loss_cls.item(), 'loss_l2':loss_l2.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward_test(self, image):
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        conf_thresh = 0.001
        nms_thresh = 0.45

        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        box_array, confs = self.detector(image)
        
        # [batch, num, 1, 4], num=16128
        box_array = box_array[:, :, 0] #.detach().cpu().numpy()
        # [batch, num, num_classes]
        # confs = confs.detach().cpu().numpy()

        # ipdb.set_trace()

        # max_conf = np.max(confs, axis=2)
        # max_id = np.argmax(confs, axis=2)

        max_conf, max_id = torch.max(confs, dim=2)
        
        b = box_array.shape[0]

        for i in range(b):
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            boxes = []
            labels = []
            scores = []

            for j in range(config.DATA.NUM_CLASSESS):

                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                # keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
                keep = nms(ll_box_array, ll_max_conf, nms_thresh)
                
                if (keep.shape[0] > 0):
                    ll_box_array = ll_box_array[keep]
                    ll_box_array = torch.clamp(ll_box_array, min=0, max=1)
                    ll_box_array[:, 0::2] *= opt.width
                    ll_box_array[:, 1::2] *= opt.height

                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    boxes.extend(ll_box_array.detach().cpu().numpy().tolist())
                    labels.extend(ll_max_id.detach().cpu().numpy().tolist())
                    scores.extend(ll_max_conf.detach().cpu().numpy().tolist())
                else:
                    pass

            batch_bboxes.append(np.array(boxes))
            batch_labels.append(np.array(labels).astype(np.int32))
            batch_scores.append(np.array(scores))

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch, published=False):
        super(Model, self).save(which_epoch, published=published)
