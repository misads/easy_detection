from torchvision.models import vgg16

def vgg16_backbone():
    backbone = vgg16(pretrained=True).features
    backbone._modules.pop('30')  # 去掉最后一层Max_Pool层

    # for layer in range(10):  # 冻结conv3之前的层
    #     for p in backbone[layer].parameters():
    #         p.requires_grad = False

    backbone.out_channels = 512
    return backbone