import os
from abc import abstractmethod

import torch
import warnings
import sys
import ipdb
from yolo3.eval_map import eval_detection_voc

from misc_utils import color_print, progress_bar
from options import opt
import misc_utils as utils
import numpy as np
from utils import deprecated
from mscv import load_checkpoint, save_checkpoint
from mscv.image import tensor2im
from mscv.aug_test import tta_inference, tta_inference_x8
from mscv.summary import write_loss


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

                batch_bboxes, batch_labels, batch_scores = self.forward(image)
                pred_bboxes.extend(batch_bboxes)
                pred_labels.extend(batch_labels)
                pred_scores.extend(batch_scores)

                for b in range(len(gt_bbox)):
                    gt_bboxes.append(gt_bbox[b].detach().cpu().numpy())
                    gt_labels.append(labels[b].int().detach().cpu().numpy())
                    gt_difficults.append(np.array([False] * len(gt_bbox[b])))

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

    # helper saving function that can be used by subclasses
    @deprecated('model.save_network() is deprecated now, use model.save() instead')
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    @deprecated('model.load_network() is deprecated now, use model.load() instead')
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            color_print("Exception: Checkpoint '%s' not found" % save_path, 1)
            if network_label == 'G':
                raise Exception("Generator must exist!,file '%s' not found" % save_path)
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path, map_location=opt.device))
                color_print('Load checkpoint from %s.' % save_path, 3)
            
            except:
                pretrained_dict = torch.load(save_path, map_location=opt.device)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

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

