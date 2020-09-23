import pdb

import numpy as np
import torch.nn as nn
from torch_template.network.norm import InsNorm
import torch.nn.functional as F
from options import opt
from .grl import GRL

import misc_utils as utils

from yolo3.darknet import Darknet

from scheduler import get_scheduler


class Domain_Classifier(nn.Module):
    def __init__(self, in_channels, middle_channels=128):
        super(Domain_Classifier, self).__init__()
        
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1),
            # 这里可能要加一个batch norm
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.domain_classifier(x)


class DA(nn.Module):
    def __init__(self, opt):
        super(DA, self).__init__()
        self.opt = opt

        cfgfile = opt.config

        self.Yolo = Darknet(cfgfile, use_cuda=True)

        self.feature = self.Yolo.common_feature

        if opt.weights:
            utils.color_print('Load Yolo weights from %s.' % opt.weights, 3)
            self.Yolo.load_weights(opt.weights)

        # self.conv1 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        # self.conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        # self.conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        # self.conv4 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        # self.conv5 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)

        """
        这里的classifier要想一个结构

        """
        in_channels = 256

        # self.domain_classifier = nn.Sequential(
        #     nn.Conv2d(in_channels, 128, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 1, kernel_size=1, stride=1),
        #     # nn.Sigmoid()
        # )

        # self.c0 = Domain_Classifier(32, 16)
        # self.c1 = Domain_Classifier(64, 32)
        # self.c2 = Domain_Classifier(128, 64)
        # self.c3 = Domain_Classifier(256, 128)
        # self.c4 = Domain_Classifier(512, 256)
        self.c5 = Domain_Classifier(1024, 512)
        

        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 2),
        # )

    # def forward(self, hazy_img, alpha=1):
    #     feature = self.feature(hazy_img) # backbone出来的feature，也可以是检测的ins_feature

    #     feature = feature[0]  # 最前面一层

    #     # feature.shape = [b, 32, 256, 256]

    #     r_feature = GRL.apply(feature, alpha)  # 梯度反向的feature
    #     detection_output = self.Yolo(hazy_img)

    #     # import ipdb; ipdb.set_trace()
    #     domain_output = self.domain_classifier(r_feature)  # 其实就是鉴别器
    #     return detection_output, domain_output


    def forward(self, hazy_img, alpha=.1):
        feature = self.feature(hazy_img) # backbone出来的feature，也可以是检测的ins_feature
        f = feature


        idxes = [2, 6, 10, 16, 20]

        # features = [f[2], f[6], f[10], f[16], f[20]]
        features = [f[20]]
        # features = [feature[idx] for idx in idxes]  # 最前面一层

        # feature.shape = [b, 32, 256, 256]

        # r_feature = GRL.apply(feature, alpha)  # 梯度反向的feature
        r_features = [GRL.apply(feature, alpha) for feature in features]
        r = r_features

        detection_output = self.Yolo(hazy_img)

        # import ipdb; ipdb.set_trace()
        # domain_output = [self.domain_classifier(r_feature) for r_feature in r_features]  # 其实就是鉴别器
        # domain_output = [self.c1(r[0]), self.c2(r[1]),self.c3(r[2]),self.c4(r[3]),self.c5(r[4])]
        domain_output = [self.c5(r[0])]
        # import ipdb; ipdb.set_trace()

        return detection_output, domain_output

