from torch import nn
import torch.nn.functional as F
from .backbone.vgg import vgg
from .box_head.box_head import SSDBoxHead


class SSDDetector(nn.Module):
    def __init__(self, opt):
        super().__init__()
        supported = [300, 512]
        if opt.scale not in supported:
            raise RuntimeError(f'image_size={opt.scale}, only support {supported}.')

        self.opt = opt
        self.backbone = vgg(image_size=opt.scale)  # backbone固定是vgg
        self.box_head = SSDBoxHead(opt)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections

