import pdb
import sys
import numpy as np
import torch
import os
from .yolo import Model as Yolo5
from torch import nn
import gc
from .eval_yolo import eval_yolo
from yolo3.eval_map import eval_detection_voc

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from mscv.summary import write_image, write_loss
from .utils import *
# from mscv.cnn import normal_init
from loss import get_default_loss

from torchvision.ops import nms

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
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        cfgfile = 'yolov5x.yaml'
        self.detector = Yolo5(cfgfile).to(opt.device)
        self.detector.hyp = hyp
        self.detector.gr = 1.0
        self.detector.nc = opt.num_classes
        #####################
        #    Init weights
        #####################
        # normal_init(self.detector)

        print_network(self.detector)

        self.optimizer = get_optimizer(opt, self.detector)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)
        self.it = 0

    def update(self, sample, *arg):
        """
        Args:
            sample: {'input': input_image [b, 3, height, width],
                   'bboxes': bboxes [b, None, 4],
                   'labels': labels [b, None],
                   'path': paths}
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

    def forward(self, sample):
        raise NotImplementedError

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        nms_thresh = 0.45  # 0.3~0.5
        conf_thresh = 0.4
        # import ipdb; ipdb.set_trace()
        round_int = lambda x: int(round(x.item())) if isinstance(x, torch.Tensor) else int(round(x))


        pred_bboxes = []
        pred_labels = []
        pred_scores = []
        gt_bboxes = []
        gt_labels = []
        gt_difficults = []

        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                utils.progress_bar(i, len(dataloader), 'Eva... ')
                
                image = sample['image'].to(opt.device)  # target domain
                gt_bbox = sample['bboxes']
                gt_label = sample['labels']

                pred = self.detector(image)
                bboxes = pred[0]
                """xywh转x1y1x2y2"""
                bboxes[:, :, 0] = bboxes[:, :, 0] - bboxes[:, :, 2] / 2
                bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes[:, :, 3] / 2
                bboxes[:, :, 2] += bboxes[:, :, 0]
                bboxes[:, :, 3] += bboxes[:, :, 1]

                # pred = non_max_suppression(pred[0], 0.4, opt.iou_thres, merge=True, classes=None, agnostic=False)

                img = image[0].detach().cpu().numpy().transpose([1, 2, 0]).copy()
                
                b = image.shape[0]  # batch有几张图
                for bi in range(b):
                    bbox = bboxes[bi]
                    conf_bbox = bbox[bbox[:, 4] > conf_thresh]
                    xyxy_bbox = conf_bbox[:, :4]  # x1y1x2y2坐标
                    scores = conf_bbox[:, 5]
                    nms_indices = nms(xyxy_bbox, scores, nms_thresh)

                    xyxy_bbox = xyxy_bbox[nms_indices]
                    scores = scores[nms_indices]
                    pred_bboxes.append(xyxy_bbox.detach().cpu().numpy())
                    pred_labels.append(np.zeros([xyxy_bbox.shape[0]], dtype=np.int32))
                    pred_scores.append(scores.detach().cpu().numpy())
                    gt_bboxes.append(gt_bbox[bi].detach().cpu().numpy())
                    gt_labels.append(np.zeros([len(gt_bbox[bi])], dtype=np.int32))
                    gt_difficults.append(np.array([False] * len(gt_bbox[bi])))

                # for x1, y1, x2, y2, *_ in nms_bbox:
                #     x1 = round_int(x1)
                #     x2 = round_int(x2)
                #     y1 = round_int(y1)
                #     y2 = round_int(y2)
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 1., 0), 2)

                # write_image(writer, 'val', f'pred{i}', img, epoch, 'HWC')

        AP = eval_detection_voc(
            pred_bboxes,
            pred_labels,
            pred_scores,
            gt_bboxes,
            gt_labels,
            gt_difficults=None,
            iou_thresh=0.5,
            use_07_metric=False)

        APs = AP['ap']
        mAP = AP['map']

        logger.info(f'Eva({data_name}) epoch {epoch}, APs: {str(APs[:opt.num_classes])}, mAP: {mAP}')
        write_loss(writer, f'val/{data_name}', 'mAP', mAP, epoch)


    def load(self, ckpt_path):
        load_dict = {
            'detector': self.detector,
        }

        if opt.resume:
            load_dict.update({
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
            })
            utils.color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            utils.color_print('Load checkpoint from %s.' % ckpt_path, 3)

        ckpt_info = load_checkpoint(load_dict, ckpt_path, map_location=opt.device)
        epoch = ckpt_info.get('epoch', 0)

        return epoch

    def save(self, which_epoch):
        save_filename = f'{which_epoch}_{opt.model}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {
            'detector': self.detector,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'epoch': which_epoch
        }

        save_checkpoint(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)



