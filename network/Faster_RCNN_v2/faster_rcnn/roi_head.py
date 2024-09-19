"""
Implements ROI_HEAD
Created by xuhaoyu.xhy
"""
import torch

from torch import nn
from torch.nn import functional as F
from torchvision.ops import MultiScaleRoIAlign, boxes as box_ops

from .box_coder import BoxCoder
from .assigner import Assigner
from .sampler import BalancedSampler


class TwoFCHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, fc_feat_channels):
        super(TwoFCHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, fc_feat_channels)
        self.fc7 = nn.Linear(fc_feat_channels, fc_feat_channels)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class BoxPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
        class_agnostic: bool, BBOX结果是否对类别敏感
    """

    def __init__(self, in_channels, num_classes, class_agnostic=False):
        super(BoxPredictor, self).__init__()
        self.class_agnostic = class_agnostic
        self.cls_score = nn.Linear(in_channels, num_classes)
        if class_agnostic:
            self.bbox_pred = nn.Linear(in_channels, 4)
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.num_classes = num_classes

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        if self.class_agnostic:
            bbox_deltas = bbox_deltas.repeat([1, self.num_classes])
        return scores, bbox_deltas


class RoIHead(torch.nn.Module):
    def __init__(self, config, in_channels=None):
        super(RoIHead, self).__init__()

        """
        Default Configs
        """
        fg_iou_thresh = config.TRAIN.ROI.FG_IOU_THRESH or 0.5
        bg_iou_thresh = config.TRAIN.ROI.BG_IOU_THRESH or 0.5
        batch_size_per_image = config.TRAIN.ROI.BATCH_SIZE or 512
        positive_fraction = config.TRAIN.ROI.POS_FRACTION or 0.25
        bbox_reg_weights = config.TRAIN.ROI.BBOX_REG_WEIGHTS or (10., 10., 5., 5.)

        self.score_thresh = config.TEST.CONF_THRESH or 0.05
        self.nms_thresh = config.TEST.NMS_THRESH or 0.5
        self.max_detections = config.TEST.MAX_DETECTIONS or 100

        num_classes = config.DATA.NUM_CLASSESS
        assert num_classes is not None, 'class names unknown.'

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=0
        )

        in_channels = in_channels or config.TRAIN.RPN.IN_CHANNELS
        assert in_channels is not None, "in_channels of RPN should be specified."

        resolution = self.box_roi_pool.output_size[0]
        fc_feat_channels = 1024
        self.box_head = TwoFCHead(
            in_channels * resolution ** 2,
            fc_feat_channels
        )
        
        class_agnostic = config.TRAIN.ROI.BBOX_CLASS_AGNOSTIC or False

        self.box_predictor = BoxPredictor(
            fc_feat_channels,
            num_classes + 1,
            class_agnostic=class_agnostic)  # 0是背景类

        self.box_coder = BoxCoder(weights=bbox_reg_weights)
        
        self.assigner = Assigner(fg_iou_thresh, bg_iou_thresh, allow_low_quality=False)
        self.sampler = BalancedSampler(batch_size_per_image, positive_fraction)


    def extra_repr(self):
        extra_str = (
            f'(box_coder): {self.box_coder}\n'
            f'(assigner): {self.assigner}\n'
            f'(sampler): {self.sampler}'
        )
        return extra_str


    def forward(self, feat_dict, proposals, image_metas, bboxes=None, labels=None):
        """
        Args:
            images: Tensor of [N, C, H, W]
            feat_dict: dict(Tensor)
            proposals: list(Tensor)
            image_metas: dict
        """
        image_shapes = image_metas['ori_sizes']

        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, bboxes, labels)

        box_features = self.box_roi_pool(feat_dict, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = {}

        proposals_concated = torch.cat(proposals, dim=0)
        boxes_per_image = [len(b) for b in proposals]
        sum_proposals = sum(boxes_per_image)

        # 返回roi预测的bbox结果
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes[:, 0, :]
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        result['pred_proposals'] = pred_boxes
        
        if self.training:
            loss_classifier, loss_box_reg = self.loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_roi_cls=loss_classifier, loss_roi_box_reg=loss_box_reg)
            result['losses'] = losses
        else:
            result.update(
                dict(class_logits=class_logits, box_regression=box_regression, proposals=proposals)
            )

        return result

    def select_training_samples(self, proposals, bboxes, labels):
        proposals = self.add_gt_proposals(proposals, bboxes)

        matched_idxs, training_labels = self.assign_targets(proposals, bboxes, labels)
        sampled_inds = self.subsample(training_labels)
        matched_gt_boxes = []
        batch_size = len(proposals)
        for i in range(batch_size):
            img_sampled_inds = sampled_inds[i]
            proposals[i] = proposals[i][img_sampled_inds]
            training_labels[i] = training_labels[i][img_sampled_inds]
            matched_idxs[i] = matched_idxs[i][img_sampled_inds]
            matched_gt_boxes.append(bboxes[i][matched_idxs[i]])

        gt_boxes_concated = torch.cat(matched_gt_boxes)
        proposals_concated = torch.cat(proposals)

        regression_targets = self.box_coder.encode(gt_boxes_concated, proposals_concated)
        return proposals, matched_idxs, training_labels, regression_targets

    def add_gt_proposals(self, proposals, bboxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, bboxes)
        ]

        return proposals

    def assign_targets(self, proposals, boxes, labels):
        matched_idxs = []
        training_labels = []
        batch_size = len(proposals)
        for i in range(batch_size):
            proposal = proposals[i]
            box = boxes[i]
            label = labels[i]

            iou_matrix = box_ops.box_iou(box, proposal)
            matched_id = self.assigner.match(iou_matrix)
            matched_id_clamped = matched_id.clamp(min=0)

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

        return matched_idxs, training_labels

    def subsample(self, labels):
        pos_masks, neg_masks = self.sampler(labels)
        sampled_inds = []
        batch_size = len(labels)
        for i in range(batch_size):
            pos_inds_img = pos_masks[i]
            neg_inds_img = neg_masks[i]

            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds


    def loss(self, class_logits, box_regression, labels, regression_targets):
        """
        Computes the loss for Faster R-CNN.

        Arguments:
            class_logits: Tensor [512 * N, NUM_CLASSESS]
            box_regression, Tensor [512 * N, NUM_CLASSESS * 4]
            labels: list(Tensor)  (N, [512]) 0是背景, 1~num_classes是类别
            regression_targets: Tensor  [512 * N, 4]

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        labels = torch.cat(labels, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        try:
            sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
            labels_pos = labels[sampled_pos_inds_subset]
            N, num_classes = class_logits.shape
            box_regression = box_regression.reshape(N, -1, 4)
        except:
            import ipdb
            ipdb.set_trace()

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        """
        说明:
            测试时的后处理

        Args:
            class_logits:   Tensor [1000*N, num_classes+1]
            box_regression: Tensor [1000*N, (num_classes+1) * 4]
            proposals:      list(Tensor) (N, [1000, 4])
            image_shapes:   list(tuple)  (N, (2, ))
        
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        batch_size = len(proposals)

        boxes_per_image = [len(b) for b in proposals]
        
        proposals_concated = torch.cat(proposals, dim=0)
        sum_proposals = sum([len(b) for b in proposals])

        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # 按图像split, 转为tuple
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        batch_boxes = []
        batch_scores = []
        batch_labels = []
        for i in range(batch_size):
            boxes = pred_boxes[i]
            scores = pred_scores[i]
            image_shape = image_shapes[i]
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.max_detections]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_labels.append(labels)

        return batch_boxes, batch_scores, batch_labels