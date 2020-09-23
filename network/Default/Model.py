import pdb
import sys
sys.path.insert(0, "./timm-efficientdet-pytorch")
sys.path.insert(0, "./omegaconf")

import numpy as np
import torch
import os

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from loss import get_default_loss

import misc_utils as utils


def get_net():
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('/home/ubuntu/.cache/torch/checkpoints/efficientdet_d5-ef44aea8.pth')
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 1024
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.detector = get_net().to(device=opt.device)
        #####################
        #    Init weights
        #####################
        # normal_init(self.detector)

        print_network(self.detector)

        self.optimizer = get_optimizer(opt, self.detector)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, data):
        """
        Args:
            data: {'input': input_image [b, 3, height, width],
                   'bboxes': bboxes [b, None, 4],
                   'labels': labels [b, None],
                   'path': paths}
        """

        # L1 & SSIM loss
        input, bboxes, labels = data['input'], data['bboxes'], data['labels']
        bboxes = [bbox.to(opt.device).float() for bbox in bboxes]
        labels = [label.to(opt.device).float() for label in labels]

        target = {'bbox': boxes, 'cls': labels}
        predicted = self.detector(input, target)

        import ipdb; ipdb.set_trace()

        loss = get_default_loss(cleaned, y, self.avg_meters)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'recovered': cleaned}

    def forward(self, x):
        return self.detector(x)

    def inference(self, x, progress_idx=None):
        return super(Model, self).inference(x, progress_idx)

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



