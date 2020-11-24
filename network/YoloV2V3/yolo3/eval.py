from __future__ import print_function
import sys
import time
import torch
from torchvision import datasets, transforms
import os
import dataset
import random
import math
import numpy as np
from utils import get_all_boxes, multi_bbox_ious, nms, read_data_cfg, logging
from cfg import parse_cfg
from darknet import Darknet
import argparse
from image import correct_yolo_boxes

# etc parameters
use_cuda      = True
seed          = 22222
eps           = 1e-5

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

FLAGS = None

def main():
    # Training settings
    datacfg       = FLAGS.data
    cfgfile       = FLAGS.config

    data_options  = read_data_cfg(datacfg)
    testlist      = data_options['valid']
    gpus          = data_options['gpus']  # e.g. 0,1,2,3
    ngpus         = len(gpus.split(','))

    num_workers   = int(data_options['num_workers'])
    # for testing, batch_size is setted to 1 (one)
    batch_size    = 1 # int(net_options['batch'])

    global use_cuda
    use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)

    ###############
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    global model
    model = Darknet(cfgfile)
    #model.print_network()

    init_width   = model.width
    init_height  = model.height

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    
    global test_loader
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist, shape=(init_width, init_height),
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]), train=False),
        batch_size=batch_size, shuffle=False, **kwargs)

    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model)
            model = model.module
    model = model.to(torch.device("cuda" if use_cuda else "cpu"))
    for w in FLAGS.weights:
        model.load_weights(w)
        logging('evaluating ... %s' % (w))
        test()

def test():
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50

    model.eval()
    num_classes = model.num_classes
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    device = torch.device("cuda" if use_cuda else "cpu")

    if model.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(model.width, model.height)
    for data, target, org_w, org_h in test_loader:
        data = data.to(device)
        output = model(data)
        all_boxes = get_all_boxes(output, shape, conf_thresh, num_classes, use_cuda=use_cuda)

        for k in range(len(all_boxes)):
            boxes = all_boxes[k]
            correct_yolo_boxes(boxes, org_w[k], org_h[k], model.width, model.height)
            boxes = np.array(nms(boxes, nms_thresh))
            truths = target[k].view(-1, 5)
            num_gts = truths_length(truths)
            total = total + num_gts
            num_pred = len(boxes)
            if num_pred == 0:
                continue

            proposals += int((boxes[:,4]>conf_thresh).sum())
            for i in range(num_gts):
                gt_boxes = torch.FloatTensor([truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                gt_boxes = gt_boxes.repeat(num_pred,1).t()
                pred_boxes = torch.FloatTensor(boxes).t()
                best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
                # pred_boxes and gt_boxes are transposed for torch.max
                if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                    correct += 1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("correct: %d, precision: %f, recall: %f, fscore: %f" % (correct, precision, recall, fscore))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',    type=str, 
        default='cfg/sketch.data', help='data definition file')
    parser.add_argument('--config', '-c',  type=str, 
        default='cfg/sketch.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w', type=str, nargs='+', 
        default=['weights/yolov3.weights'], help='initial weights file')
    FLAGS, _ = parser.parse_known_args()

    main()
