from __future__ import print_function
import sys
if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile')
    exit()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import lmdb_utils
import random
import math
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet


# Training settings
datacfg       = sys.argv[1]
cfgfile       = sys.argv[2]
weightfile    = sys.argv[3]

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

traindb       = data_options['train']
testdb        = data_options['valid']
backupdir     = data_options['backup']
nsamples      = lmdb_utils.lmdb_nsamples(traindb)

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])

max_epochs    = max_batches*batch_size/nsamples+1
use_cuda      = True
seed          = 22222
eps           = 1e-5

###############
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

model       = Darknet(cfgfile)
region_loss = model.loss

model.load_weights(weightfile)
model.print_network()
init_epoch = model.seen / nsamples 

kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    lmdb_utils.lmdbDataset(testdb, shape=(160, 160),
                   shuffle=False,
                   transform=None,
                   train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    model = torch.nn.DataParallel(model).cuda()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging('set lr=%f' % (lr))

def train(epoch):
    train_loader = torch.utils.data.DataLoader(
        lmdb_utils.lmdbDataset(traindb, shape=(model.module.width, model.module.height),
                       shuffle=True,
                       train=True, seen=model.module.seen),
        batch_size=batch_size, shuffle=False, **kwargs)

    logging('epoch %d : processed %d samples' % (epoch, epoch * len(train_loader.dataset)))
    model.train()
    adjust_learning_rate(optimizer, epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx+1) % 70 == 0:
            sys.stdout.write('.')

        if use_cuda:
            data = data.cuda()
            #target= target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = region_loss(output, target)
        loss.backward()
        optimizer.step()
    print('')
    logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
    model.module.seen = (epoch + 1) * len(train_loader.dataset)
    model.module.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    num_classes = model.module.num_classes
    anchors     = model.module.anchors
    num_anchors = model.module.num_anchors
    conf_thresh = 0.25
    nms_thresh  = 0.4
    iou_thresh  = 0.5
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        if output.size(0) == 1:
            all_boxes = [all_boxes]
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
     
            total = total + num_gts
    
            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0]
                best_iou = 0
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    best_iou = max(iou, best_iou)
                if best_iou > iou_thresh:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

evaluate = True
if evaluate:
    print('evaluating ...')
    test(0)
else:
    for epoch in range(init_epoch, max_epochs): 
        train(epoch)
        test(epoch)
