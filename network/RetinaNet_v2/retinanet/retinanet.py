"""
Implements RetinaNet
Created by xuhaoyu.xhy
"""
import torch

from torch import nn
from torchvision.ops import boxes as box_ops
from torch.nn import functional as F

from misc_utils import preview
from utils import exception
from network.__base__.backbones import get_backbone
from .fpn import FPN
from network.Faster_RCNN_v2.faster_rcnn.transforms import batch_images
from network.Faster_RCNN_v2.faster_rcnn.anchors_v2 import AnchorGenerator
from network.Faster_RCNN_v2.faster_rcnn.box_coder import BoxCoder
from network.Faster_RCNN_v2.faster_rcnn.assigner import Assigner

class RetinaNet(nn.Module):
    """
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_head (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        config (dict): config of network settings and hyper parameters
    """
    def __init__(self, config={}):
        super(RetinaNet, self).__init__()
        
        self.backbone = get_backbone(config)

        fpn_in_channels = config.TRAIN.FPN.IN_CHANNELS or [512, 1024, 2048]
        fpn_channels = config.TRAIN.FPN.OUT_CHANNELS or 256  # rpn in channels

        self.fpn = FPN(*fpn_in_channels, feature_size=fpn_channels)
        
        anchor_sizes = config.TRAIN.RPN.ANCHOR_SIZES or (32, 64, 128, 256, 512)
        anchor_aspect_ratios = config.TRAIN.RPN.ASPECT_RATIOS or (0.5, 1.0, 2.0)
        anchor_octaves = config.TRAIN.RPN.ANCHOR_OCTAVES or (1., 2**(1/3), 2**(2/3))

        num_anchors = len(anchor_aspect_ratios) * len(anchor_octaves)

        fg_iou_thresh = config.TRAIN.RPN.FG_IOU_THRESH or 0.5
        bg_iou_thresh = config.TRAIN.RPN.BG_IOU_THRESH or 0.4
        bbox_reg_weights = config.TRAIN.ROI.BBOX_REG_WEIGHTS or (10., 10., 5., 5.)
        # batch_size_per_image = config.TRAIN.RPN.BATCH_SIZE or 256
        # positive_fraction = config.TRAIN.RPN.POS_FRACTION or 0.5

        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios, anchor_octaves)

        num_classes = config.DATA.NUM_CLASSESS
        assert num_classes is not None, 'class names unknown.'

        self.regression_model = RegressionModel(fpn_channels, num_anchors)
        self.classification_model = ClassificationModel(fpn_channels, num_anchors, num_classes=num_classes + 1)

        self.box_coder = BoxCoder(weights=bbox_reg_weights)

        self.assigner = Assigner(fg_iou_thresh, bg_iou_thresh, allow_low_quality=True)
        # self.rpn = RPN(config, in_channels=fpn_channels)
        # self.roi_heads = RoIHead(config, in_channels=fpn_channels)

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
        anchors = self.anchor_generator(images, features)
        anchors_concated = torch.cat(anchors, dim=0)
        
        regression = torch.cat([self.regression_model(feature) for feature in features], dim=1)
        classification = torch.cat([self.classification_model(feature) for feature in features], dim=1)

        assert anchors[0].shape[0] == regression.shape[1]

        if self.training:
            matched_idxs, training_labels, matched_gt_boxes = self.assign_targets(anchors, bboxes, labels)
            # labels, matched_gt_boxes = self.assigner.assign_targets(anchors, bboxes)  # rpn不需要类别labels
            # torch.nonzero(labels[0] == 1).shape[0] 为正样本数

            boxes_per_image = [len(b) for b in matched_gt_boxes]
            matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors_concated).split(boxes_per_image, 0)
            # regression_targets[0][torch.nonzero(labels[0] == 1)[:, 0]] 为正样本的回归targers

            losses = self.loss(classification, regression, matched_idxs, training_labels, regression_targets)
            return losses

        return None

    def loss(self, classification, regression, matched_idxs, labels, regression_targets):
        """
        Args:
            classification: Tensor [N, num_anchors, num_classes]
            regression: Tensor [N, num_anchors, 4]
            labels: list(Tensor) (N, [num_anchors])
            regression_targets: list(Tensor) (N, [num_anchors, 4])

        """
        alpha = 0.25
        gamma = 2.0
        targets = torch.ones_like(classification) * -1  # -1为忽略，不计算loss
        classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

        box_losses = []
        num_positives = 0
        for i in range(len(labels)):
            label = labels[i]
            targets[i][label > -1] = 0.  # 前景 + 背景
  
            # positive_indices = torch.nonzero(label > 0).flatten()  # 前景     
            positive_indices = torch.nonzero(matched_idxs[i]).flatten()  # 前景     

            targets[i][positive_indices, label[positive_indices]] = 1.

            num_positives += positive_indices.numel()

            if positive_indices.numel() > 0:
                box_loss = F.smooth_l1_loss(
                    regression[i, positive_indices],
                    regression_targets[i][positive_indices]
                )
                box_losses.append(box_loss)
            else:
                box_losses.append(torch.tensor(0).float().to(regression.device))

        alpha_factor = torch.ones_like(targets) * alpha
        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

        bce = F.binary_cross_entropy(classification, targets, reduction='none')
        cls_loss = focal_weight * bce

        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))
        cls_loss = cls_loss.sum() / max(1., num_positives)
        
        box_loss = torch.stack(box_losses).mean()
        losses = dict(box_loss=box_loss, cls_loss=cls_loss)

        return losses


    def assign_targets(self, anchors, boxes, labels):
        matched_idxs = []
        training_labels = []
        matched_gt_boxes = []
        batch_size = len(anchors)
        for i in range(batch_size):
            anchor = anchors[i]
            box = boxes[i]
            label = labels[i]

            iou_matrix = box_ops.box_iou(box, anchor)
            matched_id = self.assigner.match(iou_matrix)
            matched_id_clamped = matched_id.clamp(min=0)
            matched_gt_boxes_per_image = box[matched_id_clamped]
            
            label = label[matched_id_clamped]
            label = label.to(dtype=torch.int64)

            # -1
            bg_inds = matched_id == self.assigner.BELOW_LOW_THRESHOLD
            label[bg_inds] = 0

            # -2
            ignore_inds = matched_id == self.assigner.BETWEEN_THRESHOLDS
            label[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(matched_id_clamped)
            training_labels.append(label)
            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return matched_idxs, training_labels, matched_gt_boxes


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        out = out.reshape(out.shape[0], -1, 4)
        return out


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        out2 = out2.reshape(x.shape[0], -1, self.num_classes)
        return out2