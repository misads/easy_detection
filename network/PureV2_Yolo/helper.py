import numpy as np
import torch
import os

from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
from options import opt


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


vgg = Vgg16().cuda(device=opt.device)

# utils.init_vgg16('./models/')
# vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))


def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_y


criterionBCE = nn.BCELoss()
criterionCAE = nn.L1Loss()


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        else:
            if np.random.uniform(0, 1) > 0.5:
                random_id = np.random.randint(self.pool_size, size=1)[0]
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return tmp
            else:
                return image


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, device='cpu'):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size(), device=self.device).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size(), device=self.device).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
