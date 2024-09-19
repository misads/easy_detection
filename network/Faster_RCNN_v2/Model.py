from ast import Pass
from typing import OrderedDict
import numpy as np
import torch

from torch import nn

from options import opt
from network.base_model import BaseModel

from .faster_rcnn import FasterRCNN

class Model(BaseModel):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, **kwargs)
        self.config = config
        # self._detector = faster_rcnn(config)
        self._detector = FasterRCNN(config)
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
        labels = sample['labels']
        for label in labels:
            label += 1.  # Faster RCNN的label从1开始, 0为背景

        image, bboxes, labels = sample['image'], sample['bboxes'], sample['labels']
        
        for b in range(len(image)):
            if len(bboxes[b]) == 0:  # 没有bbox，不更新参数
                return {}

        #image = image.to(opt.device)
        bboxes = [bbox.to(opt.device).float() for bbox in bboxes]
        labels = [label.to(opt.device).float() for label in labels]
        image = list(im.to(opt.device) for im in image)

        b = len(bboxes)

        # target = [{'boxes': bboxes[i], 'labels': labels[i].long()} for i in range(b)]
        """
            target['boxes'] = boxes
            target['labels'] = labels
            # target['masks'] = None
            target['image_id'] = torch.tensor([index])
            target['area'] = area
            target['iscrowd'] = iscrowd
        """
        loss_dict = self.detector(image, bboxes, labels)
        self.avg_meters.update({loss_name: loss_dict[loss_name].item() for loss_name in loss_dict})
        loss = sum(l for l in loss_dict.values())

        self.avg_meters.update({'loss': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward_test(self, sample):  # test
        """给定一个batch的图像, 输出预测的[bounding boxes, labels和scores], 仅在验证和测试时使用"""
        #image = list(im for im in image)
        image = sample['image']
        image = list(im.to(opt.device) for im in image)

        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        with torch.no_grad():
            outputs = self.detector(image)

        batch_size = len(image)

        for i in range(batch_size):
            boxes = outputs['boxes'][i]
            labels = outputs['labels'][i]
            scores = outputs['scores'][i]
            labels = labels.detach().cpu().numpy()
            # for i in range(len(labels)):
            #     labels[i] = coco_90_to_80_classes(labels[i])
            labels = labels - 1

            batch_bboxes.append(boxes.detach().cpu().numpy())
            batch_labels.append(labels)
            batch_scores.append(scores.detach().cpu().numpy())

        return batch_bboxes, batch_labels, batch_scores

    # def load(self, ckpt_path):
    #     load_dict = torch.load(ckpt_path, map_location='cpu')
    #     load_dict = load_dict['detector']

    #     self.detector.backbone._modules.pop('fc')

    #     state_dict = OrderedDict()
    #     for name in load_dict:
    #         value = load_dict[name]
    #         if name.startswith('backbone.fpn'):
    #             name = name[9:]
    #         elif name.startswith('backbone.body.'):
    #             name = 'backbone.' + name[14:]

    #         state_dict[name] = value
            
    #     self.detector.load_state_dict(state_dict)
    #     color_print('Load checkpoint from %s.' % ckpt_path, 3)

    #     return 0
