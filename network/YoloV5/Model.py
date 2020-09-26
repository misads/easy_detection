import pdb
import sys
import numpy as np
import torch
import os
from .yolo import Model as model
from torch import nn
import gc
from yolo3.darknet import Darknet
from yolo3.utils import get_all_boxes, bbox_iou, nms, read_data_cfg, load_class_names
from yolo3.image import correct_yolo_boxes
from .eval_yolo import eval_yolo

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from mscv.summary import write_image
from .utils import *
# from mscv.cnn import normal_init
from loss import get_default_loss
from utils.ensemble_boxes import non_maximum_weighted
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
        self.detector = model(cfgfile).to(opt.device)
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
        opt.iou_thres = 0.7
        conf_thresh = 0.5
        # import ipdb; ipdb.set_trace()
        round_int = lambda x: int(round(x.item())) if isinstance(x, torch.Tensor) else int(round(x))

        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                if i > 30:
                    break
                image = sample['image'].to(opt.device)  # target domain
                target = sample['yolo5_boxes'].to(opt.device)
                pred = self.detector(image)
                pred = non_max_suppression(pred[0], 0.4, opt.iou_thres, merge=True, classes=None, agnostic=False)

                img = image[0].detach().cpu().numpy().transpose([1, 2, 0]).copy()
                bbox = pred[0]

                conf_bbox = bbox[bbox[:, 4] > conf_thresh]
                # for x_c, y_c, w, h, *_ in conf_bbox:
                #     x1 = int(round((x_c - w / 2).item()))
                #     x2 = int(round((x_c + w / 2).item()))
                #     y1 = int(round((y_c - h / 2).item()))
                #     y2 = int(round((y_c + h / 2).item()))
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 1., 0), 2)
                for x1, y1, x2, y2, *_ in conf_bbox:
                    x1 = round_int(x1)
                    x2 = round_int(x2)
                    y1 = round_int(y1)
                    y2 = round_int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 1., 0), 2)

                write_image(writer, 'val', f'pred{i}', img, epoch, 'HWC')


        # eval_yolo(self.detector, dataloader, epoch, writer, logger, dataname=data_name)

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



