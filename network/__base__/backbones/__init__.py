from options.helper import is_distributed
from .resnet import resnet
from .vgg import vgg

import torch.nn as nn
import torch


def get_backbone(config):
    backbone_name = config.MODEL.BACKBONE or 'resnet50'  # 默认为resnet50
    if backbone_name in ['resnet50', 'resne101', 'resnet152']:
        # 多卡时使用 SyncBN
        kwargs = {'norm_layer': torch.nn.SyncBatchNorm} if is_distributed() else {}
        backbone = resnet(backbone_name, **kwargs)

    elif backbone_name in ['vgg16', 'vgg19']:
        backbone = vgg(backbone_name)

    return backbone
