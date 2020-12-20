import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import itertools
import struct # get_image_size
import imghdr # get_image_size

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0]-w1/2.0, box2[0]-w2/2.0)
        x2_max = max(box1[0]+w1/2.0, box2[0]+w2/2.0)
        y1_min = min(box1[1]-h1/2.0, box2[1]-h2/2.0)
        y2_max = max(box1[1]+h1/2.0, box2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea/uarea)

def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_all_boxes(output, netshape, conf_thresh, num_classes, only_objectness=1, validation=False, device='cuda:0'):
    """
    Args:
        output: 网络输出，格式如下
        output = [
            { 
                'x': 最小尺度的feature map, [batch, c, h, w]), 其中 c 等于 num_anchors * (num_classes + 5),
                    VOC为 5 × (20 + 5) = 125, COCO为 3 × (80 + 5) = 255,
                    h 和 w 为这一级别的feature高和宽, 输入为416×416时, Yolov2 为 13×13
                    Yolov3分别为 13×13

                'a': anchor大小, [N * 2] Tensor 类型
                'n': anchor数目, int类型
            },
            {
                'x': 第二个尺度(×2)网络的feature map, 输入为416×416时, 为26×26
                'a':
                'n':
            },
            {
                'x': 第三个尺度(×4)网络的feature map, 输入为416×416时, 为52×52
                'a':
                'n':
            },
            ....

        ]

    Returns:
        all_boxes: [batch, N, 5 + num_classes] Tensor, N是bbox的数量, 
            5 + num_classes+ 是每个bbox的回归坐标、置信度以及每一类的分类预测结果

    """
    batch = output[0]['x'].shape[0]
    all_boxes = torch.tensor([]).view([batch, 0, num_classes+5]).to(device)

    # output[0]是最小的尺寸，anchors最少 [1]和[2]越来越大
    for i in range(len(output)):
        pred = output[i]['x'].data

        # find number of workers (.s.t, number of GPUS) 
        nw = output[i]['n'].data.size(0)
        anchors = output[i]['a'].chunk(nw)[0]
        num_anchors = output[i]['n'].data[0].item()

        b = get_region_boxes(
            pred, 
            netshape, 
            conf_thresh, 
            num_classes, 
            anchors, 
            num_anchors,
            only_objectness=only_objectness, 
            validation=validation, 
            device=device
        )
        all_boxes = torch.cat([all_boxes, b], dim=1)

    return all_boxes


def get_region_boxes(output, netshape, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False, device='cuda:0'):
    # device = torch.device("cuda" if use_cuda else "cpu")
    anchors = anchors.to(device)
    anchor_step = anchors.size(0)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    cls_anchor_dim = batch*num_anchors*h*w
    if netshape[0] != 0:
        nw, nh = netshape
    else:
        nw, nh = w, h

    t0 = time.time()
    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, cls_anchor_dim)

    grid_x = torch.linspace(0, w-1, w).repeat(batch*num_anchors, h, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(cls_anchor_dim).to(device)
    ix = torch.LongTensor(range(0,2)).to(device)
    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(batch, h*w).view(cls_anchor_dim)
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(batch, h*w).view(cls_anchor_dim)

    xs, ys = output[0].sigmoid() + grid_x, output[1].sigmoid() + grid_y
    ws, hs = output[2].exp() * anchor_w.detach(), output[3].exp() * anchor_h.detach()
    det_confs = output[4].sigmoid()

    # by ysyun, dim=1 means input is 2D or even dimension else dim=0
    cls_confs = torch.nn.Softmax(dim=1)(output[5:5+num_classes].transpose(0,1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    
    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors

    xs = (xs.view([batch, -1]) / w).unsqueeze(2)
    ys = (ys.view([batch, -1]) / h).unsqueeze(2)
    ws = (ws.view([batch, -1]) / nw).unsqueeze(2)
    hs = (hs.view([batch, -1]) / nh).unsqueeze(2)

    det_confs = det_confs.view([batch, -1]).unsqueeze(2)
    cls_confs = cls_confs.view([batch, h * w * num_anchors, -1])
    all_boxes = torch.cat([xs,ys,ws,hs,det_confs,cls_confs],2)
    return all_boxes

 