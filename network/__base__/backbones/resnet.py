from typing import OrderedDict
from torchvision.ops import misc as misc_nn_ops
# from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models.resnet import Bottleneck, Any, ResNet, load_state_dict_from_url, model_urls
import torch

class ResNet_Features(ResNet):
    # 把4层特征全部返回
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)
        x = torch.flatten(x, 1)

        outputs = OrderedDict()
        outputs[0] = layer1
        outputs[1] = layer2
        outputs[2] = layer3
        outputs[3] = layer4
        return outputs

def _resnet(
    arch: str,
    block,
    layers,
    pretrained: bool,
    progress: bool,
    **kwargs,
):
    model = ResNet_Features(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


backbone_map = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}

def resnet(backbone_name, 
           forzen_stage=1, 
           pretrained=True,
           norm_layer=misc_nn_ops.FrozenBatchNorm2d):

    assert backbone_name in backbone_map
    backbone = backbone_map[backbone_name](pretrained=pretrained, norm_layer=norm_layer)

    if forzen_stage:
        for name, parameter in backbone.named_parameters():
            froze_flag = True
            for i in range(forzen_stage + 1, 5):  # 一共4层layer
                if f'layer{i}' in name:
                    froze_flag = False

            if froze_flag:
                parameter.requires_grad_(False)

    return backbone