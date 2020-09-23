import pdb

import numpy as np
import torch
import os
import tensorboardX
from torch import optim
from torch import nn 

import misc_utils as utils
from torch_template.network.base_model import BaseModel
from torch_template.network.metrics import ssim, L1_loss
from torch_template.utils.torch_utils import ExponentialMovingAverage, print_network

from options import opt
from yolo3.darknet import Darknet
from yolo3.utils import get_all_boxes, bbox_iou, nms, read_data_cfg, load_class_names
from yolo3.image import correct_yolo_boxes

from scheduler import get_scheduler

from .helper import ImagePool, GANLoss, criterionCAE, gradient, vgg
from .PureV2 import cleaner
from .grl import GRL
from .domain_adaptation import DA

import ipdb
import gc

criterion = nn.CrossEntropyLoss()
criterionBCE = nn.BCELoss()

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.DA = DA(opt)

        utils.color_print('DA:', 2)
        print_network(self.DA)

        if opt.optimizer == 'adam':
            # 只优化cleaner common_features 是固定死的
            self.optimizer = optim.Adam(self.DA.Yolo.parameters(), lr=opt.lr, betas=(0.95, 0.999))
        elif opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.DA.Yolo.parameters(), 
                        lr=0.001/opt.batch_size, momentum=0.9, 
                        dampening=0, weight_decay=0.0005*opt.batch_size)
            # self.optimizer = optim.SGD(self.DA.Yolo.parameters(), lr=opt.lr)

        self.scheduler = get_scheduler(opt, self.optimizer)
        
        # load pre-trained networks
        if opt.load:
            pretrained_path = opt.load
            self.load_network(self.DA, 'G', opt.which_epoch, pretrained_path)
        
        # if opt.weights:
        #     utils.color_print('Load Yolo weights from %s.' % opt.weights, 3)
        #     self.Yolo.load_weights(opt.weights)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)


    def detect(self, hazy_img):

        conf_thresh = 0.005
        nms_thresh = 0.45
        self.Yolo.eval()
        if self.Yolo.net_name() == 'region': # region_layer
            shape=(0, 0)
        else:
            shape=(self.Yolo.width, self.Yolo.height)

        output = self.Yolo(hazy_img)
        batch_boxes = get_all_boxes(output, shape, conf_thresh, self.Yolo.num_classes, only_objectness=0, validation=True)

        self.Yolo.train()

        for i in range(len(batch_boxes)):
            # lineId += 1
            # fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            #width, height = get_image_size(valid_files[lineId])

            # 这里是crop 怎么detect呢？
            ipdb.set_trace()
            width, height = hazy_img.size(2), hazy_img.size(3)  # 这一句可能有点问题
            boxes = batch_boxes[i]
            correct_yolo_boxes(boxes, width, height, self.Yolo.width, self.Yolo.height)
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                for j in range((len(box)-5)//2):
                    cls_conf = box[5+2*j]
                    cls_id = int(box[6+2*j])
                    prob = det_conf * cls_conf
                    ipdb.set_trace()
                    # fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))

        return 

    def update(self, x, target=None, tgt_img=None, alpha=1):  # x是有雾霾图, clean是无雾图

        self.optimizer.zero_grad()
        model = self.DA
        
        detection_output, src_domain_outputs = model(x, alpha=alpha)
        _, tgt_domain_outputs = model(tgt_img, alpha=alpha)

        # 准备domain分类器的label
        # label_src = torch.zeros_like(src_domain_output).to(opt.device)  # source 0
        # label_tgt = torch.ones_like(tgt_domain_output).to(opt.device)

        loss_layers = model.Yolo.loss_layers
        org_loss = []
        src_domain_loss = []

        for src_domain_output in src_domain_outputs:
            label_src = torch.zeros_like(src_domain_output).to(opt.device) 
            src_domain_loss.append(criterionBCE(src_domain_output, label_src))

        tgt_domain_loss = []
        for tgt_domain_output in tgt_domain_outputs:
            label_tgt = torch.ones_like(tgt_domain_output).to(opt.device)
            tgt_domain_loss.append(criterionBCE(tgt_domain_output, label_tgt))

        for i, l in enumerate(loss_layers):
            # l.seen = l.seen + data.data.size(0)
            ol=l(detection_output[i]['x'], target)
            org_loss.append(ol)
        
        # 检测loss 两个域分类loss
        detection_loss = sum(org_loss)
        src_domain_loss_sum = sum(src_domain_loss) * 20  # 非常小
        tgt_domain_loss_sum = sum(tgt_domain_loss) * 20 # 非常大

        self.avg_meters.update({'detection': detection_loss.item()})
        self.avg_meters.update({'src': src_domain_loss_sum.item()})
        self.avg_meters.update({'tgt': tgt_domain_loss_sum.item()})

        loss = detection_loss + src_domain_loss_sum + tgt_domain_loss_sum

        loss.backward()

        nn.utils.clip_grad_norm_(model.Yolo.parameters(), 10000)
        self.optimizer.step()

        org_loss.clear()
        src_domain_loss.clear()
        tgt_domain_loss.clear()
        gc.collect()
        # """
        # alpha的值要改一下
        # """
        # alpha = 1.0  
        # # train on source domain
        # src_class_output, src_domain_output = self.DA(x, alpha=alpha)
        # src_loss_detection = criterion(src_class_output, detection_src)  # 优化下游任务
        # src_loss_domain = criterion(src_domain_output, label_src)  # 源域的label全为0

        # # train on target domain
        # _, tgt_domain_output = self.DA(images_tgt, alpha=alpha)
        # tgt_loss_domain = criterion(tgt_domain_output, label_tgt)  # 目标域的label全为1

        # # 总的损失 = 下游任务(检测、分类、分割等等)的损失 + 域差别损失
        # loss = src_loss_detection + src_loss_domain + tgt_loss_domain

        # # optimize dann
        # loss.backward()
        # self.optimizer.step()

        return {'recovered': None}

    # def forward(self, x, alpha=1):
    #     # detection_output, domain_output
    #     return self.DA(x, alpha)

    def inference(self, x, image=None):
        pass

    def save(self, which_epoch):
        self.save_network(self.DA, 'G', which_epoch)
        self.DA.Yolo.save_weights(os.path.join(self.save_dir, '%06d.weights' % which_epoch))

    def adjust_learning_rate(self, batch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = 0.001
        steps = [-1,500,40000,60000]
        scales = [0.1,10,.1,.1]
        for i in range(len(steps)):
            scale = scales[i] if i < len(scales) else 1
            if batch >= steps[i]:
                lr = lr * scale
                if batch == steps[i]:
                    break
            else:
                break
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr/opt.batch_size
        return lr