from torchvision.models import vgg16
from torchvision.models import resnet101
from torchvision.ops import misc as misc_nn_ops
from torch import nn

def vgg16_backbone():
    backbone = vgg16(pretrained=True).features
    backbone._modules.pop('30')  # 去掉最后一层Max_Pool层

    # for layer in range(10):  # 冻结conv3之前的层
    #     for p in backbone[layer].parameters():
    #         p.requires_grad = False
    #         p.requires_grad_(False)

    backbone.out_channels = 512
    return backbone

def res101_backbone():
    res = resnet101(pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)


    # for name, parameter in backbone.named_parameters():  # 冻结layer2之前的层
    #     if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
    #         parameter.requires_grad_(False)

    backbone = nn.Sequential(*[res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3, res.layer4])

    backbone.out_channels = 2048
    return backbone