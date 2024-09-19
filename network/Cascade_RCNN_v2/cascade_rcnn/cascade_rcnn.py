"""
Implements Cascade R-CNN
Created by xuhaoyu.xhy
"""
import torch

from options.helper import is_distributed, is_first_gpu

from torch import nn

from misc_utils import preview
from utils import exception, EasyDict, AverageMeters, SumMeters
from network.__base__.backbones import get_backbone
from network.__base__.necks import FPN

from network.Faster_RCNN_v2.faster_rcnn.transforms import batch_images
from network.Faster_RCNN_v2.faster_rcnn.rpn import RPN
from network.Faster_RCNN_v2.faster_rcnn.roi_head import RoIHead

class CascadeRCNN(nn.Module):
    """
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_head (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        config (dict): config of network settings and hyper parameters
    """
    def __init__(self, config={}):
        super(CascadeRCNN, self).__init__()
        
        self.backbone = get_backbone(config)

        fpn_in_channels = config.TRAIN.FPN.IN_CHANNELS or [256, 512, 1024, 2048]
        fpn_channels = config.TRAIN.FPN.OUT_CHANNELS or 256  # rpn in channels

        self.fpn = FPN(fpn_in_channels, fpn_channels, extra_blocks='pool')
        
        self.rpn = RPN(config, in_channels=fpn_channels)
        self.roi_heads = nn.ModuleList()

        fg_iou_thresh = config.TRAIN.ROI.FG_IOU_THRESH or (0.5, 0.6, 0.7)
        bg_iou_thresh = config.TRAIN.ROI.BG_IOU_THRESH or (0.5, 0.6, 0.7)
        
        self.loss_weights = config.TRAIN.ROI.LOSS_WEIGHTS or (1.0, 0.5, 0.25)
        
        cascade_bbox_reg_weights = [(10., 10., 5., 5.),  (20., 20., 10., 10.), (30., 30., 15., 15.)]
        bbox_reg_weights = config.TRAIN.ROI.BBOX_REG_WEIGHTS or cascade_bbox_reg_weights

        assert isinstance(fg_iou_thresh, tuple) or isinstance(fg_iou_thresh, list)
        assert isinstance(bg_iou_thresh, tuple) or isinstance(bg_iou_thresh, list)

        self.num_stages = len(fg_iou_thresh)
        for i in range(self.num_stages):
            config.TRAIN.ROI.FG_IOU_THRESH = fg_iou_thresh[i]
            config.TRAIN.ROI.BG_IOU_THRESH = bg_iou_thresh[i]
            config.TRAIN.ROI.BBOX_REG_WEIGHTS = bbox_reg_weights[i]
            config.TRAIN.ROI.BBOX_CLASS_AGNOSTIC = True
            self.roi_heads.append(RoIHead(config, in_channels=fpn_channels))

        self.config = config


    def forward(self, images, bboxes=None, labels=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            bboxes (list[Tensor]): a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]]
            labels: (list[Tensor]): a list of labels [[N1], [N2], ..., [Nb]]

        Returns:
            During training:
                dict[Tensor] which contains the losses.
                dict(losses=losses)

            During testing:
                dict[Tensor] which contains the detection result.
                {
                    'boxes': list(Tensor),
                    'labels': list(Tensor),
                    'scores' list(Tensor)
                }

        """
        without_targets = bboxes is None or labels is None
        if self.training and without_targets:
            exception('In training mode, targets should be passed')

        ori_sizes = [tuple(img.shape[-2:]) for img in images]
        images = batch_images(images)  # collect images to batch
        image_metas = dict(ori_sizes=ori_sizes)

        feat_dict = self.backbone(images)
        if self.fpn is not None:
            feat_dict = self.fpn(feat_dict)

        # 转为list
        if isinstance(feat_dict, torch.Tensor):
            feat_dict = {0: feat_dict}

        features = list(feat_dict.values())  

        rpn_result = self.rpn(images, features, image_metas, bboxes, labels)

        proposals = rpn_result['proposals']  # list(Tensor)  (N, [-1, 4])
        rpn_logits = rpn_result['logits']
        # dict(str->Tensor)

        roi_weighted_losses = SumMeters()  # (训练时) 多个cascade head求加权loss
        class_logits_mean = AverageMeters()  # (测试时) class_logits求平均

        for i in range(self.num_stages):
            roi_result = self.roi_heads[i](feat_dict, proposals, image_metas, bboxes, labels)
            proposals = roi_result['pred_proposals']
            if self.training:
                roi_losses = roi_result['losses']
                roi_weighted_losses.update(roi_losses, weight=self.loss_weights[i])
            else:
                class_logits = roi_result['class_logits']
                class_logits_mean.update(dict(class_logits=class_logits))

        if self.training:
            losses = {}
            rpn_losses = rpn_result['losses']
            losses.update(rpn_losses)
            losses.update(roi_weighted_losses)
            return losses
        else:
            # eval
            class_logits = class_logits_mean['class_logits']
            box_regression = roi_result['box_regression']
            proposals = roi_result['proposals']
            batch_boxes, batch_scores, batch_labels = self.roi_heads[-1].postprocess_detections(class_logits, box_regression, proposals, ori_sizes)

            return dict(boxes=batch_boxes, scores=batch_scores, labels=batch_labels)
        

# def cascade_rcnn(config):
#     backbone_name = config.MODEL.BACKBONE or 'resnet50'  # 默认为resnet50

#     kwargs = {}
#     if backbone_name in ['resnet50', 'resne101', 'resnet152']:
#         # 多卡时使用 SyncBN
#         kwargs = {'norm_layer': torch.nn.SyncBatchNorm} if is_distributed() else {}
#         resnet_backbone = resnet(backbone_name, **kwargs)
#         backbone = fpn_backbone(resnet_backbone)
#         model = CascadeRCNN(backbone, config)

#     elif backbone_name in ['vgg16', 'vgg19']:
#         backbone = vgg(backbone_name)
#         model = CascadeRCNN(backbone, config)

#     else:
#         exception(f'no such backbone: {backbone_name}')

#     return model
