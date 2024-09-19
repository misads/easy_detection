"""
Implements the R-CNN series framework
Created by xuhaoyu.xhy
"""
import torch

from torch import nn

from misc_utils import preview
from utils import exception
from network.__base__.backbones import get_backbone
from network.__base__.necks import FPN
from network.Faster_RCNN.frcnn.rpn import RegionProposalNetwork
from .transforms import batch_images
from .rpn import RPN
from network.Faster_RCNN.frcnn.rpn import AnchorGenerator
from .roi_head import RoIHead

class FasterRCNN(nn.Module):
    """
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_head (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        config (dict): config of network settings and hyper parameters
    """
    def __init__(self, config={}):
        super(FasterRCNN, self).__init__()
        
        self.backbone = get_backbone(config)

        fpn_in_channels = config.TRAIN.FPN.IN_CHANNELS or [256, 512, 1024, 2048]
        fpn_channels = config.TRAIN.FPN.OUT_CHANNELS or 256  # rpn in channels

        self.fpn = FPN(fpn_in_channels, fpn_channels, extra_blocks='pool')
        
        self.rpn = RPN(config, in_channels=fpn_channels)
        self.roi_heads = RoIHead(config, in_channels=fpn_channels)

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

        roi_result = self.roi_heads(feat_dict, proposals, image_metas, bboxes, labels)

        if self.training:
            losses = {}
            rpn_losses = rpn_result['losses']
            roi_losses = roi_result['losses']

            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        # eval
        class_logits = roi_result['class_logits']
        box_regression = roi_result['box_regression']
        proposals = roi_result['proposals']
        batch_boxes, batch_scores, batch_labels = self.roi_heads.postprocess_detections(class_logits, box_regression, proposals, ori_sizes)

        return dict(boxes=batch_boxes, scores=batch_scores, labels=batch_labels)


# def faster_rcnn(config):
#     backbone_name = config.MODEL.BACKBONE or 'resnet50'  # 默认为resnet50

#     kwargs = {}
#     if backbone_name in ['resnet50', 'resne101', 'resnet152']:
#         # 多卡时使用 SyncBN
#         kwargs = {'norm_layer': torch.nn.SyncBatchNorm} if is_distributed() else {}
#         resnet_backbone = resnet(backbone_name, **kwargs)
#         backbone = fpn_backbone(resnet_backbone)
#         model = FasterRCNN(backbone, config)

#     elif backbone_name in ['vgg16', 'vgg19']:
#         backbone = vgg(backbone_name)
#         model = FasterRCNN(backbone, config)

#     else:
#         exception(f'no such backbone: {backbone_name}')

#     return model
