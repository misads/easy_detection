"""
Implements RPN
Created by xuhaoyu.xhy
"""
import torch

from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from .anchors_v2 import AnchorGenerator
from .box_coder import BoxCoder
from .assigner import Assigner
from .sampler import BalancedSampler


class RPNHead(nn.Module):
    """
    分类: 卷积(in_channels -> in_channels) + 卷积(in_channels -> num_anchors_ratios)
    回归: 卷积(in_channels -> in_channels) + 卷积(in_channels -> num_anchors_ratios * 4)

    Args:
        in_channels (int): 输入通道数, 通常是FPN的通道数
        num_anchors_ratios (int): 不同比例的anchor数

    Return:
        logits: list(Tensor)
        bbox_reg: list(Tensor)
        对应每层feature的logit和bbox_reg

            logits[0]: Tensor [B, num_anchors_ratios, feat_height, feat_width]
            bbox_reg[0]: Tensor [B, num_anchors_ratios * 4, feat_height, feat_width]
            
    """

    def __init__(self, in_channels, num_anchors_ratios):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors_ratios, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors_ratios * 4, kernel_size=1, stride=1)

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPN(torch.nn.Module):
    def __init__(self, config, in_channels=None):
        super(RPN, self).__init__()

        """
        Default Configs
        """
        anchor_sizes = config.TRAIN.RPN.ANCHOR_SIZES or (32, 64, 128, 256, 512)
        anchor_aspect_ratios = config.TRAIN.RPN.ASPECT_RATIOS or (0.5, 1.0, 2.0)
        anchor_octaves = config.TRAIN.RPN.ANCHOR_OCTAVES or (1,)

        fg_iou_thresh = config.TRAIN.RPN.FG_IOU_THRESH or 0.7
        bg_iou_thresh = config.TRAIN.RPN.BG_IOU_THRESH or 0.3
        batch_size_per_image = config.TRAIN.RPN.BATCH_SIZE or 256
        positive_fraction = config.TRAIN.RPN.POS_FRACTION or 0.5

        self.nms_thresh = config.TRAIN.RPN.NMS_THRESH or 0.7
        self.min_size = 1e-3

        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios, anchor_octaves)

        in_channels = in_channels or config.TRAIN.RPN.IN_CHANNELS
        assert in_channels is not None, "in_channels of RPN should be specified."

        self.rpn_head = RPNHead(in_channels, len(anchor_aspect_ratios) * len(anchor_octaves))

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.assigner = Assigner(fg_iou_thresh, bg_iou_thresh, allow_low_quality=True)
        self.sampler = BalancedSampler(batch_size_per_image, positive_fraction)
        self.config = config


    @property
    def pre_nms_top_n(self):
        if self.training:
            return self.config.TRAIN.RPN.NMS_PRE or 2000
        else:
            return self.config.TEST.RPN.NMS_PRE or 1000

    @property
    def post_nms_top_n(self):
        if self.training:
            return self.config.TRAIN.RPN.NMS_POST or 2000
        else:
            return self.config.TEST.RPN.NMS_POST or 1000


    def forward(self, images, features, image_metas, bboxes=None, labels=None):
        """
        Args:
            images: Tensor of [N, C, H, W]
            features: list(Tensor)
            image_metas: dict
        """
        logits, bbox_reg = self.rpn_head(features) 
        anchors = self.anchor_generator(images, features)

        batch_size = len(anchors)

        num_anchors = anchors[0].shape[0]
        num_anchors_per_level = [logit[0].numel() for logit in logits]  # 一个logit对应一个anchor

        assert sum(num_anchors_per_level) == num_anchors, "Num of anchors does not match with logits."

        """
        Flatten logits and bbox_reg
        """
        num_feature_levels = len(logits)
        for i in range(num_feature_levels):
            """
            因为anchors生成为[num_shifts, num_ratios, 4].reshape(-1, 4)
            所以将feature转为 H, W 在前, Channels(num_anchor_ratios) 在后
            """
            N, C, H, W = logits[i].shape
            # logits[i].shape = [N, num_anchor_ratios, H, W]
            logits[i] = logits[i].view(N, -1, H, W).permute(0, 2, 3, 1).reshape(N, -1, 1)
            # bbox_reg[i].shape = [N, num_anchor_ratios * 4, H, W]
            bbox_reg[i] = bbox_reg[i].view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)

        logits = torch.cat(logits, dim=1).reshape(-1, 1)
        bbox_reg = torch.cat(bbox_reg, dim=1).reshape(-1, 4)

        anchors_concated = torch.cat(anchors, dim=0)  # 将一个batch的anchors cancat到一起
        # boxes_per_image = [len(b) for b in boxes]
        sum_anchors = sum([len(b) for b in anchors])
        proposals = self.box_coder.decode(bbox_reg.detach(), anchors)

        proposals = proposals.view(batch_size, -1, 4)

        batch_boxes, batch_logits = self.filter_proposals(
            proposals, logits, image_metas['ori_sizes'], num_anchors_per_level)

        losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assigner.assign_targets(anchors, bboxes)  # rpn不需要类别labels
            # torch.nonzero(labels[0] == 1).shape[0] 为正样本数

            boxes_per_image = [len(b) for b in matched_gt_boxes]
            matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors_concated).split(boxes_per_image, 0)
            # regression_targets[0][torch.nonzero(labels[0] == 1)[:, 0]] 为正样本的回归targers

            rpn_losses = self.loss(logits, bbox_reg, labels, regression_targets)
            losses.update(rpn_losses)

            return dict(proposals=batch_boxes, logits=batch_logits, losses=losses)
        else:  # test
            return dict(proposals=batch_boxes, logits=batch_logits, losses={})
    

    def filter_proposals(self, proposals, logits, image_shapes, num_anchors_per_level):
        """
        说明:
            每个feat_level保留置信度最高的top_n个anchors
            裁剪图像外的bbox坐标, 去除面积过小的bbox

        Args:
            proposals: Tensor [N, -1, 4]
            logits: Tensor [-1, 1]
            image_shapes: list(tuple)
            num_anchors_per_level: list(int)

        Returns:
            batch_boxes: list(Tensor) (N, [post_nms_top_n, 4])
            batch_scores: list(Tensor) (N, [post_nms_top_n])
        """
        batch_size = len(proposals)
        logits = logits.detach()
        logits = logits.reshape(batch_size, -1)

        device = proposals.device
        levels = []
        for i, num_anchors in enumerate(num_anchors_per_level):
            levels += [i] * num_anchors   # 0, 0, 0, 1, 1, ....., 4, 4, 4
                
        levels = torch.as_tensor(levels, dtype=torch.int64, device=device)
        
        assert len(levels) == logits.shape[1]  
        levels = levels.reshape(1, -1).expand_as(logits)  # levels.shape == logits.shape

        # 每个feat_level保留置信度最高的top_n个anchors
        top_n_idx = self._get_top_n_idx(logits, num_anchors_per_level, self.pre_nms_top_n)
        batch_idx = torch.arange(batch_size, device=device)[:, None]

        logits = logits[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        batch_boxes = []
        batch_scores = []

        for i in range(batch_size):
            boxes = proposals[i]
            scores = logits[i]
            level = levels[i]
            image_shape = image_shapes[i]
            # 裁剪超出图像范围的bbox
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # 去除面积过小的bbox
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, level = boxes[keep], scores[keep], level[keep]

            # 对于每个level分别做nms
            keep = box_ops.batched_nms(boxes, scores, level, self.nms_thresh)
            # 保留最大的n个结果
            keep = keep[:self.post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]

            batch_boxes.append(boxes)
            batch_scores.append(scores)

        return batch_boxes, batch_scores
        
    @staticmethod
    def _get_top_n_idx(logits, num_anchors_per_level, top_n):
        """
        说明:
            每个feat level获取top n个置信度最高的anchors, 返回它们的id

        Args:
            logits: Tensor [B, num_anchors]
            num_anchors_per_level: list(int)
            top_n: int 保留的个数

        Returns:
            idx: Tensor [B, filtered_num_anchors]
        """
        assert logits.shape[1] == sum(num_anchors_per_level)

        idx = []
        offset = 0
        for i, num_anchors in enumerate(num_anchors_per_level):
            top_k = min(top_n, num_anchors)
            logit = logits[:, offset: offset + num_anchors]
            # 每个level取top_n个, 如果没有top_n个anchors, 则全取
            values, indices = torch.topk(logit, top_k)

            idx.append(indices + offset)
            offset += num_anchors

        idx = torch.cat(idx, dim=1)  # Tensor [B, -1]
        return idx

    def loss(self, logits, bbox_reg, labels, regression_targets):
        pos_masks, neg_masks = self.sampler(labels)  # list(tensor)  (N, [len(anchors_per_image)])

        sampled_pos_inds = torch.nonzero(torch.cat(pos_masks, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(neg_masks, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        logits = logits.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.l1_loss(
            bbox_reg[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            logits[sampled_inds], labels[sampled_inds]
        )

        losses = {
            "loss_rpn_cls": objectness_loss,
            "loss_rpn_box_reg": box_loss,
        }
        return losses