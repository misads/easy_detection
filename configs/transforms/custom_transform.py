from __future__ import absolute_import, division

import math
import random
import numbers
import warnings
from enum import IntEnum
from types import LambdaType
from typing import Optional

import cv2
import numpy as np
from skimage.measure import label

# opencv-python >= 4.2.0.34
# opencv-python-headless >= 4.2.0.34
# albumentations >= 0.5.1
from albumentations import functional as F
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform, NoOp, to_tuple
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox, union_of_bboxes


class RandomPad(DualTransform):
    def __init__(
        self,
        min_height: Optional[int] = 1024,
        min_width: Optional[int] = 1024,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        super(RandomPad, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(RandomPad, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]


        if rows < self.min_height:
            # h_pad_top = int((self.min_height - rows) / 2.0)
            h_pad_top = random.randint(0, self.min_height - rows)
            h_pad_bottom = self.min_height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.min_width:
            # w_pad_left = int((self.min_width - cols) / 2.0)
            w_pad_left = random.randint(0, self.min_width - cols)
            w_pad_right = self.min_width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0


        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.value
        )

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.mask_value
        )

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    # skipcq: PYL-W0613
    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "border_mode",
            "value",
            "mask_value",
        )


class RandomResize(DualTransform):
    def __init__(self, scale_limit, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(RandomResize, self).__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        h, w, _  = img.shape

        ratio = random.uniform(1 / (1 + self.scale_limit), 1 + self.scale_limit)
        height = h
        width = int(ratio * w)

        return F.resize(img, height=height, width=width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("scale_limit", "interpolation")