import torch
from torch.nn import functional as F
from torch import nn

from torchvision.ops import boxes as box_ops

class Assigner:

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, fg_iou_thresh, bg_iou_thresh, allow_low_quality=True):
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.allow_low_quality = allow_low_quality

    def assign_targets(self, anchors, gt_bboxes):
        """
        说明:
            根据anchors和gt_bboxes分配正样本和负样本(背景)

        Args:
            anchors: list(Tensor)  每张图像的anchors
            gt_bboxes:  list(Tensor)  每张图像的bboxes
            gt_labels:  list(Tensor)  每张图像的labels

        Returns:
            labels: list(Tensor) (N, [len(anchors_per_image)]) 取值为1, 0或-1
                 1:     正样本(iou>0.7(如果没有, 则取iou最大的))
                 0:     背景(负样本, iou<0.3)
                -1:     丢弃(0.3<=iou<0.7)

            matched_gt_boxes: List(Tensor) (N, [len(anchors_per_image), 4])
                每个 anchor 对应的与之匹配的gt_bboxes坐标, 仅当labels == 1时有效
            
        """
        batch_size = len(anchors)
        labels = []
        matched_gt_boxes = []
        for i in range(batch_size):
            anchors_per_image = anchors[i]
            bboxes = gt_bboxes[i]

            iou_matrix = box_ops.box_iou(bboxes, anchors_per_image) # Tensor [len(bboxes), len(anchors_per_image)]
            matched_idxs = self.match(iou_matrix)
            
            matched_gt_boxes_per_image = bboxes[matched_idxs.clamp(min=0)]
            
            labels_per_image = torch.where(matched_idxs >= 0, 1., 0.) # 1是正样本, 0是背景, -1是丢弃(iou在中间)
 
            # 背景 (负样本)
            labels_per_image[matched_idxs == self.BELOW_LOW_THRESHOLD] = 0

            # 去除iou在0.3~0.7之间的下标(不是正样本也不是负样本)
            labels_per_image[matched_idxs == self.BETWEEN_THRESHOLDS] = -1

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        
        return labels, matched_gt_boxes

    def match(self, iou_matrix):
        """
        说明:
            根据iou_matrix返回哪些anchors时正样本(有目标)

        Args:
            iou_matrix: Tensor [len(bboxes), len(anchors_per_image)]

        Returns:
            match_idx: Tensor [len(anchors_per_image)], 返回每个anchor匹配的bbox id, 背景则mask为-1, 非背景和前景mask为-2

        """
        assert iou_matrix.numel() != 0, "No ground-truth boxes available for one of the images during training"
        values, match_idx = iou_matrix.max(dim=0)

        if self.allow_low_quality:
            match_idx_backup = match_idx.clone()

        # 将背景部分mask掉
        match_idx[values < self.bg_iou_thresh] = self.BELOW_LOW_THRESHOLD
        match_idx[(values >= self.bg_iou_thresh) & (values < self.fg_iou_thresh)] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality:
            self.low_quality_matches(match_idx, match_idx_backup, iou_matrix)

        return match_idx

    def low_quality_matches(self, match_idx, match_idx_backup, iou_matrix):
        """
        说明:
            当 allow_low_quality=True 时, 每个gt_bbox都获取至少一个anchor与之对应, 即使iou < 0.7。
            如果多个anchors与这个gt_bbox的iou值相同, 这些anchors都认为可以匹配。
        """
        highest_quality_foreach_gt, _ = iou_matrix.max(dim=1)

        gt_pred_pairs_of_highest_quality = torch.nonzero(
            iou_matrix == highest_quality_foreach_gt[:, None]
        )
        #           gt_id anchor_id
        # tensor([[     0, 234268],
        #         [     0, 234271],
        #         [     0, 234274],
        #         [     0, 234277],
        #         [     0, 234280],
        #         [     0, 234496],
        #         [     0, 234499],
        #         [     1, 234251],
        #         [     1, 234254],
        #         [     2, 200348],
        #         [     3, 200323],
        #         [     3, 200779]], device='cuda:0')
        recover = gt_pred_pairs_of_highest_quality[:, 1]
        match_idx[recover] = match_idx_backup[recover]

    def __repr__(self):
        return (f'{self.__class__.__name__}'
            f'(fg_iou_thresh={self.fg_iou_thresh}, '
            f'bg_iou_thresh={self.bg_iou_thresh}, '
            f'allow_low_quality={self.allow_low_quality})')