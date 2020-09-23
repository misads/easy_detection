import pdb

import numpy as np
import torch.nn as nn
from torch_template.network.norm import InsNorm
import torch.nn.functional as F
from options import opt


class cleaner(nn.Module):
    def __init__(self):
        super(cleaner, self).__init__()

        # Initial convolutional layers
        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)  # `SAME` padding
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # DuRBs, a DualUpDownLayer is a DuRB_US
        self.rud1 = DualUpDownLayer(64, 64, 64, f_sizes=(3, 5), dilation=1, norm_type='batch_norm')
        self.rud2 = DualUpDownLayer(64, 64, 64, f_sizes=(3, 5), dilation=1, norm_type='batch_norm')
        self.rud3 = DualUpDownLayer(64, 64, 64, f_sizes=(5, 7), dilation=1, norm_type='batch_norm')

        # Last layers
        # -- Up1 --
        self.upconv1 = ConvLayer(64, 256, kernel_size=3, stride=1)
        self.upsamp1 = nn.PixelShuffle(2)
        # ----------
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)

        # -- Up2 --
        self.upconv2 = ConvLayer(64, 256, kernel_size=3, stride=1)
        self.upsamp2 = nn.PixelShuffle(2)
        # ----------
        self.se = SEBasicBlock(64, 64, reduction=32)
        self.pa = PALayer(64)
        # ----------
        self.conv5 = ConvLayer(64, 64, kernel_size=3, stride=1)

        self.end_conv = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        _, _, pad_h, pad_w = input.size()
        if pad_h % 8 != 0 or pad_w % 8 != 0:
            h_pad_len = 8 - pad_h % 8
            w_pad_len = 8 - pad_w % 8

            input = F.pad(input, (0, w_pad_len, 0, h_pad_len), mode='reflect')

        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        blue = x  # big size
        x = self.relu(self.conv3(x))  # small size

        x, blue = self.rud1(x, blue)
        x, blue = self.rud2(x, blue)
        x, _ = self.rud3(x, blue)

        x = self.upconv1(x)
        x = self.upsamp1(x)
        x = self.relu(self.conv4(x))

        x = self.upconv2(x)
        x = self.upsamp2(x)

        x = self.se(x)  # channel attention
        x = self.pa(x)  # pixel attention

        x = self.relu(self.conv5(x))

        x = self.tanh(self.end_conv(x))
        # x = x + input  # 用feature 不太好加

        x = x[:, :, :pad_h, :pad_w]
        return x


class F_Block(nn.Module):
    def __init__(self, dim, f_sizes=(3, 5), dilation=1, norm="instance"):
        super(F_Block, self).__init__()

        self.conv1_up = ConvLayer(dim, dim, kernel_size=(f_sizes[0]), stride=1, dilation=dilation)
        self.conv1_down = ConvLayer(dim, dim, kernel_size=(f_sizes[1]), stride=1, dilation=dilation)

        self.conv2_up = ConvLayer(dim, dim, kernel_size=(f_sizes[0]), stride=1, dilation=dilation)
        self.conv2_down = ConvLayer(dim, dim, kernel_size=(f_sizes[1]), stride=1, dilation=dilation)

        self.relu = nn.ReLU()

    def forward(self, input):
        x = input
        a_up = self.conv1_up(x)
        a_down = self.conv1_down(x)
        mid = self.relu(a_up + a_down)

        b_up = self.conv2_up(mid)
        b_down = self.conv2_down(mid)
        x = self.relu(a_up + a_down + b_up + b_down)

        return x + input


G_Block = F_Block


# DualUpDownLayer is DuRB_US, defined here:
class DualUpDownLayer(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, f_sizes=(3, 5), dilation=1, norm_type="instance", with_relu=True):
        super(DualUpDownLayer, self).__init__()

        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)

        # T^{l}_{1}: (up+conv.)
        # -- Up --

        self.conv_pre = ConvLayer(in_dim, 4*in_dim, 3, 1)
        self.upsamp = nn.PixelShuffle(2)
        # --------
        # self.up_conv = ConvLayer(res_dim, res_dim, kernel_size=f_size, stride=1, dilation=dilation)
        # --------
        self.f_block = F_Block(res_dim, f_sizes=f_sizes, dilation=dilation)
        self.g_block = G_Block(res_dim, f_sizes=f_sizes, dilation=dilation)

        # T^{l}_{2}: (se+conv.), stride=2 for down-scaling.
        self.se = SEBasicBlock(res_dim, res_dim, reduction=32)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=3, stride=2)
        # self.down_conv = ConvLayer(res_dim, res_dim, kernel_size=f_size, stride=2)

        self.with_relu = with_relu
        if opt.norm is not None:
            self.bn1 = Norm(opt.norm, out_dim)

        self.relu = nn.ReLU()

        if opt.norm is not None:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, input, blue):

        x = self.relu(self.conv1(input))
        x = self.conv2(x)
        x += input
        x = self.relu(x)  # [8 64 64 64]

        x = self.conv_pre(x)  # [8 256 64 64]
        x = self.upsamp(x)  # [8 64 128 128]
        # x = self.up_conv(x)  # [8 64 128 128]
        x = self.f_block(x)

        x += blue
        x = self.relu(x)
        blue = x

        x = self.se(x)
        # x = self.down_conv(x)  # new added

        x = self.down_conv(x)  # [8 64 64 64]
        x = self.g_block(x)

        x += input

        if opt.norm is not None:
            x = self.bn1(x)

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        if opt.norm is not None:
            # x = F.dropout(x, p=0.5, inplace=False, training=self.training)
            x = self.dropout(x)

        return x, blue

    # ------------------------------------


class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2
        self.reflection_pad = nn.ReflectionPad2d(self.padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

        # if dilation == 1:
        #     reflect_padding = int(np.floor(kernel_size / 2))
        #     self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        #     self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)
        # else:
        #     self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, padding=dilation)

    def forward(self, x):

        # if self.dilation == 1:
        if self.padding > 0:
            out = self.reflection_pad(x)
        else:
            out = x
        out = self.conv2d(out)
        # else:
        #     out = self.conv2d(x)
        return out


class Norm(nn.Module):
    def __init__(self, norm_type, dim):
        super(Norm, self).__init__()
        if norm_type is None:
            self.norm = lambda x: x
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(dim)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise Exception("Normalization type incorrect. Valid: {batch|instance}.")

    def forward(self, x):
        out = self.norm(x)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# SE-ResNet Module
class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=64, with_norm=False):
        super(SEBasicBlock, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes, 1)
        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU(inplace=True)
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.se(out)
        out += x
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=64):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
                nn.Linear(reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y
