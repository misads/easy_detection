import os
from abc import abstractmethod

import cv2
import torch
import warnings
import sys
import ipdb
from utils.eval_metrics.eval_map import eval_detection_voc

from misc_utils import color_print, progress_bar
from options import opt
import misc_utils as utils
import numpy as np
from utils import deprecated
from mscv import load_checkpoint, save_checkpoint
from mscv.image import tensor2im
from mscv.aug_test import tta_inference, tta_inference_x8
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im
from utils.vis import visualize_boxes

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def eval_mAP(self, dataloader, epoch, writer, logger, data_name='val'):
        # eval_yolo(self.detector, dataloader, epoch, writer, logger, dataname=data_name)
        pred_bboxes = []
        pred_labels = []
        pred_scores = []
        gt_bboxes = []
        gt_labels = []
        gt_difficults = []

        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                utils.progress_bar(i, len(dataloader), 'Eva... ')
                image = sample['image'].to(opt.device)
                gt_bbox = sample['bboxes']
                labels = sample['labels']
                paths = sample['path']

                batch_bboxes, batch_labels, batch_scores = self.forward(image)
                pred_bboxes.extend(batch_bboxes)
                pred_labels.extend(batch_labels)
                pred_scores.extend(batch_scores)

                for b in range(len(gt_bbox)):
                    gt_bboxes.append(gt_bbox[b].detach().cpu().numpy())
                    gt_labels.append(labels[b].int().detach().cpu().numpy())
                    gt_difficults.append(np.array([False] * len(gt_bbox[b])))

                if opt.vis:  # 可视化预测结果
                    img = tensor2im(image).copy()
                    # for x1, y1, x2, y2 in gt_bbox[0]:
                    #     cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)  # 绿色的是gt

                    num = len(batch_scores[0])
                    visualize_boxes(image=img, boxes=batch_bboxes[0],
                             labels=batch_labels[0].astype(np.int32), probs=batch_scores[0], class_labels=opt.class_names)

                    write_image(writer, f'{data_name}/{i}', 'image', img, epoch, 'HWC')

            result = []
            for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                AP = eval_detection_voc(
                    pred_bboxes,
                    pred_labels,
                    pred_scores,
                    gt_bboxes,
                    gt_labels,
                    gt_difficults=None,
                    iou_thresh=iou_thresh,
                    use_07_metric=False)

                APs = AP['ap']
                mAP = AP['map']
                result.append(mAP)

                logger.info(f'Eva({data_name}) epoch {epoch}, IoU: {iou_thresh}, APs: {str(APs[:opt.num_classes])}, mAP: {mAP}')

                write_loss(writer, f'val/{data_name}', 'mAP', mAP, epoch)

            logger.info(
                f'Eva({data_name}) epoch {epoch}, mean of (AP50-AP75): {sum(result)/len(result)}')


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

        s = torch.load(ckpt_path)
        if opt.resume:
            self.optimizer.load_state_dict(s['optimizer'])

        epoch = ckpt_info.get('epoch', 0)

        return epoch

    def save(self, which_epoch, published=False):
        save_filename = f'{which_epoch}_{opt.model}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {
            'detector': self.detector,
            'epoch': which_epoch
        }
        
        if published:
            save_dict['epoch'] = 0
        else:
            save_dict['optimizer'] = self.optimizer
            save_dict['scheduler'] = self.scheduler

        save_checkpoint(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)

