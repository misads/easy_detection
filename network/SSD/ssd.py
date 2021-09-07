from torch import nn
import torch.nn.functional as F
from .backbone.vgg import vgg
from .box_head.box_head import SSDBoxHead


class SSDDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        supported = [300, 512]
        if config.DATA.SCALE not in supported:
            raise RuntimeError(f'image_size={config.DATA.SCALE}, only support {supported}.')

        self.config = config
        self.backbone = vgg(image_size=config.DATA.SCALE)  # backbone固定是vgg
        self.box_head = SSDBoxHead(config)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections

