import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
from options import opt

criterionCAE = nn.L1Loss()

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):  # (1) relu1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):  # (3) relu1_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 7):  # (6) relu2_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 9):  # (8) relu2_2
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 12):  # (11) relu3_1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        relu1_1 = self.slice1(X)
        relu1_2 = self.slice2(relu1_1)
        relu2_1 = self.slice3(relu1_2)
        relu2_2 = self.slice4(relu2_1)
        relu3_1 = self.slice5(relu2_2)
        out = [relu1_1, relu1_2, relu2_1, relu2_2, relu3_1]
        return out


vgg = Vgg16().to(device=opt.device)


def vgg_loss(recovered, label):
    features_est = vgg(recovered)  # [relu1_1, relu1_2, relu2_1, relu2_2, relu3_1]
    features_gt = vgg(label)

    content_loss = []

    for i in range(4):  # [relu1_1, relu1_2, relu2_1, relu2_2]
        content_loss.append(criterionCAE(features_est[i], features_gt[i]))

    content_loss = sum(content_loss)
    return content_loss