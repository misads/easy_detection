from torchvision.models import vgg16, vgg19
from torchvision.models import resnet101
from torchvision.ops import misc as misc_nn_ops
from torch import nn

backbone_map = {
    'vgg16': vgg16,
    'vgg19': vgg19
}

def vgg(backbone_name, 
        forzen_layers=10, 
        pretrained=True):

    assert backbone_name in backbone_map
    backbone = vgg16(pretrained=True).features
    backbone._modules.pop('30')  # 去掉最后一层Max_Pool层

    for layer in range(forzen_layers):  # 冻结conv3之前的层
        for p in backbone[layer].parameters():
            p.requires_grad_(False)

    backbone.out_channels = 512
    return backbone
