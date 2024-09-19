from __future__ import division

import math
import torch

class BoxCoder(object):
    """
    训练时框回归的目标是中心坐标偏移(dx, dy)以及宽高偏移log(dw)和log(dh)
    bbox通常格式为(x1, y1, x2, y2)

    decode:
        将 bbox_reg 和 anchors 解码为 bbox 坐标

    encode:
        将 gt_bbox 和 anchors 编码为回归目标值

        解码后再编码与原始值一样 encode(decode(bbox_reg)) = bbox_reg
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    @staticmethod
    def xyxy_to_cxcywh(bboxes):
        """
        说明:
            左上右下坐标转为中心、宽高
            (x1, y1, x2, y2) 转为 (cx, cy, w, h)

        Args:
            bboxes: Tensor [N, 4]
        """
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        cx = bboxes[:, 0] + 0.5 * widths
        cy = bboxes[:, 1] + 0.5 * heights
        return cx, cy, widths, heights


    @staticmethod
    def cxcywh_to_xyxy(cx, cy, w, h):
        """
        说明:
            中心、宽高坐标转为左上右下坐标转
             (cx, cy, w, h)转为(x1, y1, x2, y2) 

        Args:
            cx: Tensor [N, 1]
            cy: Tensor [N, 1]
            w:  Tensor [N, 1]
            h:  Tensor [N, 1]
        """

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return x1, y1, x2, y2


    def encode(self, gt_boxes, anchors):
        """
        说明:
             将 gt_bbox 和 anchors(或者rpn proposals) 编码为回归目标值

        Args:
            gt_boxes: Tensor [N, 4]
            anchors: Tensor [N, 4]
        """
        assert isinstance(gt_boxes, torch.Tensor)
        assert isinstance(anchors, torch.Tensor), "anchors should be concated to tensor first before bbox decode."

        # dtype = gt_boxes.dtype
        # device = gt_boxes.device
        # weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        wx, wy, ww, wh = self.weights

        gt_cx, gt_cy, gt_widths, gt_heights = self.xyxy_to_cxcywh(gt_boxes)
        anchor_cx, anchor_cy, anchor_widths, anchor_heights = self.xyxy_to_cxcywh(anchors)
        
        targets_dx = wx * (gt_cx - anchor_cx) / anchor_widths
        targets_dy = wy * (gt_cy - anchor_cy) / anchor_heights
        targets_dw = ww * torch.log(gt_widths / anchor_widths)
        targets_dh = wh * torch.log(gt_heights / anchor_heights)

        targets = torch.cat((targets_dx[:, None], targets_dy[:, None], targets_dw[:, None], targets_dh[:, None]), dim=1)
        return targets
        
    def decode(self, rel_codes, boxes):
        assert isinstance(boxes, (list, tuple))
        if isinstance(rel_codes, (list, tuple)):
            rel_codes = torch.cat(rel_codes, dim=0)
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [len(b) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        pred_boxes = self.decode_single(
            rel_codes.reshape(sum(boxes_per_image), -1), concat_boxes
        )
        return pred_boxes.reshape(sum(boxes_per_image), -1, 4)


    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


    def decode_v2(self, bbog_reg, anchors, sum_boxes):
        """
        说明:
            将 bbox_reg 和 anchors 解码为 bbox 坐标

        Args:
            bbog_reg: Tensor [N, 4], 框回归值, (cx, cy, dw*, dh*) 格式, dw*和dh*是取了log后的偏移值
            anchors: Tensor [N, 4], 一个batch里concat后的所有anchors
            sum_boxes: int, 所有的boxes数量, 等于batch中每张图像的bbox数量之和

        """
        assert isinstance(bbog_reg, torch.Tensor)
        assert isinstance(anchors, torch.Tensor), "anchors should be concated to tensor first before bbox decode."

        # if len(bbog_reg.shape) != 2:
        #     bbog_reg = bbog_reg.reshape(-1, 4)

        bbog_reg = bbog_reg.reshape(sum_boxes, -1)

        rel_codes = bbog_reg
        anchors = anchors.type_as(rel_codes)

        anchor_cx, anchor_cy, anchor_w, anchor_h = self.xyxy_to_cxcywh(anchors)

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # pred bbox
        pred_cx = dx * anchor_w[:, None] + anchor_cx[:, None]
        pred_cy = dy * anchor_h[:, None] + anchor_cy[:, None]
        pred_w = torch.exp(dw) * anchor_w[:, None]
        pred_h = torch.exp(dh) * anchor_h[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_cx - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_cy - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_cx + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_cy + 0.5 * pred_h

        return pred_boxes.reshape(sum_boxes, -1, 4)

    def __repr__(self):
        return f'{self.__class__.__name__}(weights={self.weights})'