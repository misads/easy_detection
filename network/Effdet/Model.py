import pdb
import sys
#sys.path.insert(0, "./timm-efficientdet-pytorch")
# sys.path.insert(0, "./omegaconf")

import numpy as np
import torch
import os

from .effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from .effdet.efficientdet import HeadNet

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init

import misc_utils as utils


def get_net(pretrained=True):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    if pretrained:
        checkpoint = torch.load('/home/ubuntu/.cache/torch/checkpoints/efficientdet_d5-ef44aea8.pth')
        net.load_state_dict(checkpoint)
    config.num_classes = config.DATA.NUM_CLASSESS
    config.image_size = opt.scale
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)


class Model(BaseModel):
    def __init__(self, opt, logger=None):
        super(Model, self).__init__(config, kwargs)
        self.opt = opt
        self.detector = get_net().to(device=opt.device)
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
        Args:
            给定一个batch的图像和gt, 更新网络权重, 仅在训练时使用.
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        loss = self.forward_train(sample)

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward_train(self, sample):
        labels = sample['labels']
        for i in range(len(labels)):
            labels[i] += 1   # effdet的label从1开始

        image, bboxes, labels = sample['image'], sample['bboxes'], sample['labels']

        for bbox in bboxes:
            bbox[:,[0,1,2,3]] = bbox[:,[1,0,3,2]]  # yxyx

        image = image.to(opt.device)
        bboxes = [bbox.to(opt.device).float() for bbox in bboxes]
        labels = [label.to(opt.device).float() for label in labels]

        # import ipdb
        # ipdb.set_trace()
        loss, _, _ = self.detector(image, bboxes, labels)

        return loss

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        total_loss = 0.
        for i, sample in enumerate(dataloader):
            utils.progress_bar(i, len(dataloader), 'Eva... ')

            with torch.no_grad():
                loss = self.forward(sample)

            total_loss += loss.item()

        logger.info(f'Eva({data_name}) epoch {epoch}, total loss: {total_loss}.')

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)

