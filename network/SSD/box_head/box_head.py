from torch import nn
import torch.nn.functional as F

from .box_predictor import make_box_predictor
from ..anchors.prior_box import PriorBox
from .inference import PostProcessor
from .loss import MultiBoxLoss
from ..utils import box_utils


CENTER_VARIANCE = 0.1
SIZE_VARIANCE = 0.2


class SSDBoxHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.predictor = make_box_predictor(config)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=3)
        self.post_processor = PostProcessor(config)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.config)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, CENTER_VARIANCE, SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}
