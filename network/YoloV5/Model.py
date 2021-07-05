import pdb
import sys
import numpy as np
import torch
import os
from .yolo import Model as Yolo5
from torch import nn
import gc
import ipdb

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from mscv.summary import write_image, write_loss
from .utils import *
# from mscv.cnn import normal_init

from torchvision.ops import nms
from .utils import non_max_suppression
from utils.ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion

import misc_utils as utils

hyp = {'momentum': 0.937,  # SGD momentum
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)

class Model(BaseModel):
    def __init__(self, opt, logger=None):
        super(Model, self).__init__(config, kwargs)
        self.opt = opt
        cfgfile = 'configs/yolov5x.yaml'
        self.detector = Yolo5(cfgfile)
        self.detector.hyp = hyp
        self.detector.gr = 1.0
        self.detector.nc = config.DATA.NUM_CLASSESS
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
        self.it = 0

    def update(self, sample, *arg):
        """
        给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        # self.it += 1
        # ni = self.it + self.nb * self.epoch
        #
        # if ni <= self.n_burn:
        #     xi = [0, self.n_burn]  # x interp
        #     # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
        #     accumulate = max(1, np.interp(ni, xi, [1, 64 / opt.batch_size]).round())
        #     for j, x in enumerate(self.optimizer.param_groups):
        #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        #         x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * (((1 + math.cos(self.epoch * math.pi / 300)) / 2) ** 1.0) * 0.9 + 0.1])
        #         if 'momentum' in x:
        #             x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

        image = sample['image'].to(opt.device)  # target domain
        target = sample['yolo5_boxes'].to(opt.device)
        pred = self.detector(image)
        loss, loss_items = compute_loss(pred, target.to(opt.device), self.detector)
        self.avg_meters.update({'loss': sum(loss_items).item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward_test(self, image):  # test
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        inf_out, _ = self.detector(image)
        
        bboxes = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.65, merge=False)

        b = len(bboxes)
        for bi in range(b):
            pred = bboxes[bi]
            if pred is None:
                batch_bboxes.append(np.array([[]]))
                batch_labels.append(np.array([]))
                batch_scores.append(np.array([]))
            else:
                boxes = pred[:,:4].cpu().detach().numpy()
                scores = pred[:,4].cpu().detach().numpy()
                labels = pred[:,5].cpu().detach().numpy().astype(np.int32)
                batch_bboxes.append(boxes)
                batch_labels.append(labels)
                batch_scores.append(scores)

        # ipdb.set_trace()
        """xywh转x1y1x2y2"""
        # bboxes[:, :, 0] = bboxes[:, :, 0] - bboxes[:, :, 2] / 2
        # bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes[:, :, 3] / 2
        # bboxes[:, :, 2] += bboxes[:, :, 0]
        # bboxes[:, :, 3] += bboxes[:, :, 1]

        # b = image.shape[0]  # batch有几张图
        # for bi in range(b):
        #     bbox = bboxes[bi]
        #     conf_bbox = bbox[bbox[:, 4] > opt.conf_thresh]
        #     xyxy_bbox = conf_bbox[:, :4]  # x1y1x2y2坐标
        #     scores = conf_bbox[:, 4]

        #     nms_indices = nms(xyxy_bbox, scores, opt.nms_thresh)

        #     xyxy_bbox = xyxy_bbox[nms_indices]
        #     scores = scores[nms_indices]  # 检测的置信度
        #     classification = conf_bbox[nms_indices, 5:]

            

        #     if len(classification) != 0:
        #         prob, class_id = torch.max(classification, 1)
        #         # scores = scores * prob  # 乘以最高类别的置信度
        #     else:
        #         class_id = torch.Tensor([])

        #     if opt.box_fusion == 'wbf':
        #         pass
        #         # boxes, scores, labels = weighted_boxes_fusion([xyxy_bbox.detach().cpu().numpy()], 
        #         #                                             [scores.detach().cpu().numpy()], 
        #         #                                             [np.zeros([xyxy_bbox.shape[0]], dtype=np.int32)], 
        #         #                                             iou_thr=0.5)
        #     elif opt.box_fusion == 'nms':
        #         boxes = xyxy_bbox.detach().cpu().numpy()
        #         scores = scores.detach().cpu().numpy()
        #         labels = class_id.detach().cpu().numpy()

        #     batch_bboxes.append(boxes)
        #     batch_labels.append(labels)
        #     batch_scores.append(scores)

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        # state = torch.load(ckpt_path)
        # utils.p(list(state['detector'].keys()))
        # print('=========================================')
        # utils.p(list(self.detector.state_dict().keys()))
        # ipdb.set_trace()
        # self.detector.load_state_dict(state)
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)


