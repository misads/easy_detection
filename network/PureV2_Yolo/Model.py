import pdb

import numpy as np
import torch
import os
import tensorboardX
from torch import optim

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
from .Discriminator import D
import ipdb


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.cleaner = cleaner().cuda(device=opt.device)
        cfgfile = opt.config
        # self.Yolo = Darknet(cfgfile, use_cuda=True)
        # Yolo是整体 前两个是部分
        self.Yolo = Darknet(cfgfile, use_cuda=True)
        self.common_feature = self.Yolo.common_feature
        # self.detector = self.Yolo.detector

        # object去雾和整体去雾使用相同的结构
        self.object_cleaner = self.cleaner  

        # self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids != '-1' else torch.Tensor
        #####################
        #    Init weights
        #####################
        # self.cleaner.apply(weights_init)

        utils.color_print('net_G:', 2)
        print_network(self.cleaner)

        if opt.optimizer == 'adam':
            # 只优化cleaner common_features 是固定死的
            self.g_optimizer = optim.Adam(self.cleaner.parameters(), lr=opt.lr, betas=(0.95, 0.999))
        elif opt.optimizer == 'sgd':
            self.g_optimizer = optim.SGD(self.cleaner.parameters(), lr=opt.lr)

        self.scheduler = get_scheduler(opt, self.g_optimizer)
        
        # load pre-trained networks
        if opt.load:
            pretrained_path = opt.load
            self.load_network(self.cleaner, 'G', opt.which_epoch, pretrained_path)
            # if self.training:
            #     self.load_network(self.discriminitor, 'D', opt.which_epoch, pretrained_path)
        
        # 加载Yolo的weights
        
        if opt.weights:
            utils.color_print('Load Yolo weights from %s.' % opt.weights, 3)
            self.Yolo.load_weights(opt.weights)

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

    def dehaze_background(self, hazy_img):
        feature = self.common_feature(hazy_img)
        return self.cleaner(feature)
      	
    def dehaze_object(self, hazy_img, bbox):
        feature = self.common_feature(hazy_img)
        object_feature = get_object_feature(feature, bbox)
        return self.object_cleaner(object_feature)
      
    def dehaze(self, hazy_img):
        background = self.dehaze_background(hazy_img)
        bboxes = self.detect(hazy_img)
        objects = []
        for bbox in bboxes:
            objects.append(self.dehaze_object(hazy_img, bbox))
            
        return self.fusion(background, objects)     

    def calc_loss_G(self, cleaned, y):
        opt = self.opt
        # cleaned = self.cleaner(x)
        ssim_loss_r = -ssim(cleaned, y)
        ssim_loss = ssim_loss_r * opt.weight_ssim  # 1.1

        # Compute L1 loss
        l1_loss = L1_loss(cleaned, y)
        l1_loss = l1_loss * opt.weight_l1  # 0.75

        loss = ssim_loss + l1_loss

        # Compute Gradient loss
        if opt.weight_grad:
            gradie_h_est, gradie_v_est = gradient(cleaned)
            gradie_h_gt, gradie_v_gt = gradient(y)

            L_tran_h = criterionCAE(gradie_h_est, gradie_h_gt)
            L_tran_v = criterionCAE(gradie_v_est, gradie_v_gt)

            loss_grad = (L_tran_h + L_tran_v) * opt.weight_grad
            loss += loss_grad

            self.avg_meters.update({'gradient': loss_grad.item()})

        self.avg_meters.update({'ssim': -ssim_loss_r.item(), 'L1': l1_loss.item()})

        # Compute vgg loss(content loss)
        if opt.weight_vgg:  # vgg loss
            features_est = vgg(cleaned)  # [relu1_1, relu1_2, relu2_1, relu2_2, relu3_1]
            features_gt = vgg(y)

            content_loss = criterionCAE(features_est[0], features_gt[0])

            for i in range(1, 4):  # [relu1_1, relu1_2, relu2_1, relu2_2]
                content_loss += criterionCAE(features_est[i], features_gt[i])

            content_loss *= opt.weight_vgg
            loss += content_loss

            self.avg_meters.update({'vgg': content_loss.item()})

        return loss, cleaned

    def update(self, x, clear_label=None):  # x是有雾霾图, clean是无雾图

        loss = []

        self.detect(x)

        common_feature = self.common_feature(x)[0]  # [1, 32, 256, 256]
        cleaned = self.cleaner(common_feature)

        loss_G, cleaned = self.calc_loss_G(cleaned, clear_label)  # 去雾的损失
        loss.append(loss_G)

        self.g_optimizer.zero_grad()  # 这里只更新cleaner，不更新detector
        sum(loss).backward()
        self.g_optimizer.step()
        loss.clear()

        return {'recovered': cleaned}

    def forward(self, x):
        return self.dehaze(x)

    def inference(self, x, image=None):
        pass

    def save(self, which_epoch):
        self.save_network(self.cleaner, 'G', which_epoch)
        self.Yolo.save_weights(os.path.join(self.save_dir, '%06d.weights' % which_epoch))

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
