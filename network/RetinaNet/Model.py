import pdb
import sys
sys.path.insert(0, "./timm-efficientdet-pytorch")
sys.path.insert(0, "./omegaconf")

import numpy as np
import torch
import cv2
import os

from torch import nn
import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .retinanet import resnet50 as Retina_50
from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from mscv.summary import write_image
from loss import get_default_loss

import misc_utils as utils


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        # cfgfile = 'yolo-voc.cfg'
        # self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        #
        # # replace the pre-trained head with a new one
        # self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.num_classes + 1)
        self.detector = Retina_50(opt.num_classes,pretrained=True)

        #####################
        #    Init weights
        #####################
        # normal_init(self.detector)

        print_network(self.detector)

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
        images = sample['image'].to(opt.device)


        annotations = [torch.cat([box, label.unsqueeze(1)], dim=1).to(opt.device) for box, label in zip(sample['bboxes'],sample['labels'])]
        inputs = images, annotations

        loss = sum(self.detector(inputs))


        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward(self, image):  # test
        conf_thresh = 0.5

        batch_scores, batch_labels, batch_bboxes = self.detector(image)
        conf = batch_scores > conf_thresh
        batch_bboxes = batch_bboxes[conf].detach().cpu().numpy()
        batch_labels = batch_labels[conf].detach().cpu().numpy()
        batch_scores = batch_scores[conf].detach().cpu().numpy()

        return [batch_bboxes], [batch_labels], [batch_scores]

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)
