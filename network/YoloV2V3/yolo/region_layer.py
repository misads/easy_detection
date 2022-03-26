import math
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import bbox_iou, multi_bbox_ious, convert2cpu

class RegionLayer(nn.Module):
    def __init__(self, num_classes=0, anchors=[1.0], num_anchors=1, device=None):
        super(RegionLayer, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)//num_anchors
        #self.anchors = torch.stack(torch.FloatTensor(anchors).split(self.anchor_step)).to(self.device)
        self.anchors = torch.FloatTensor(anchors).view(self.num_anchors, self.anchor_step).to(self.device)
        self.rescore = 1
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def build_targets(self, pred_boxes, target, nH, nW):
        nB = target.size(0)
        nA = self.num_anchors
        noobj_mask = torch.ones (nB, nA, nH, nW)
        obj_mask   = torch.zeros(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        tcoord     = torch.zeros( 4, nB, nA, nH, nW)
        tconf      = torch.zeros(nB, nA, nH, nW)
        tcls       = torch.zeros(nB, nA, nH, nW)

        nAnchors = nA*nH*nW
        nPixels  = nH*nW
        nGT = 0 # number of ground truth
        nRecall = 0
        # it works faster on CPU than on GPU.
        anchors = self.anchors.to("cpu")

        if self.seen < 12800:
            tcoord[0].fill_(0.5)
            tcoord[1].fill_(0.5)
            coord_mask.fill_(0.01)
            # initial w, h == 0 means log(1)==0, s.t, anchor is equal to ground truth.

        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1,5).to("cpu")
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gw = [ i * nW for i in (tbox[t][1], tbox[t][3]) ]
                gy, gh = [ i * nH for i in (tbox[t][2], tbox[t][4]) ]
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors,1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = (cur_ious>self.thresh).view(nA,nH,nW)
            noobj_mask[b][ignore_ix] = 0

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gw = [ i * nW for i in (tbox[t][1], tbox[t][3]) ]
                gy, gh = [ i * nH for i in (tbox[t][2], tbox[t][4]) ]
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)

                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA,1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, 2), anchors),1).t()
                tmp_ious = multi_bbox_ious(anchor_boxes, tmp_gt_boxes, x1y1x2y2=False)
                best_iou, best_n = torch.max(tmp_ious, 0)

                if self.anchor_step == 4: # this part is not tested.
                    tmp_ious_mask = (tmp_ious==best_iou)
                    if tmp_ious_mask.sum() > 0:
                        gt_pos = torch.FloatTensor([gi, gj, gx, gy]).repeat(nA,1).t()
                        an_pos = anchor_boxes[4:6] # anchor_boxes are consisted of [0 0 aw ah ax ay]
                        dist = pow(((gt_pos[0]+an_pos[0])-gt_pos[2]),2) + pow(((gt_pos[1]+an_pos[1])-gt_pos[3]),2)
                        dist[1-tmp_ious_mask]=10000 # set the large number for the small ious
                        _, best_n = torch.min(dist,0)

                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)

                try:
                    if gj >= nH:
                        gj = nH - 1
                    if gi >= nW:
                        gi = nW - 1

                    obj_mask  [b][best_n][gj][gi] = 1
                    noobj_mask[b][best_n][gj][gi] = 0
                    coord_mask[b][best_n][gj][gi] = 2. - tbox[t][3]*tbox[t][4]
                    tcoord [0][b][best_n][gj][gi] = gx - gi
                    tcoord [1][b][best_n][gj][gi] = gy - gj
                    tcoord [2][b][best_n][gj][gi] = math.log(gw/anchors[best_n][0])
                    tcoord [3][b][best_n][gj][gi] = math.log(gh/anchors[best_n][1])
                    tcls      [b][best_n][gj][gi] = tbox[t][0]
                    tconf     [b][best_n][gj][gi] = iou if self.rescore else 1.
                    if iou > 0.5:
                        nRecall += 1
                except Exception as e:
                    print('obj_mask', obj_mask.shape)
                    print(f'b: {b}')
                    print(f'best_n: {best_n}')
                    print(f'nB: {nB}')
                    print(f'gj: {gj}')
                    print(f'gi: {gi}')
                    print(str(e))
                    raise Exception('Error')
        return nGT, nRecall, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls

    def get_mask_boxes(self, output):
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step).to(self.device)
        masked_anchors = self.anchors.view(-1)
        num_anchors = torch.IntTensor([self.num_anchors]).to(self.device)
        return {'x':output, 'a':masked_anchors, 'n':num_anchors}

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)    # batch size
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls_anchor_dim = nB*nA*nH*nW

        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step).to(self.device)

        output = output.view(nB, nA, (5+nC), nH, nW).to(self.device)
        cls_grid = torch.linspace(5,5+nC-1,nC).long().to(self.device)
        ix = torch.LongTensor(range(0,5)).to(self.device)
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to(self.device)

        coord = output.index_select(2, ix[0:4]).view(nB*nA, -1, nH*nW).transpose(0,1).contiguous().view(-1,cls_anchor_dim)  # x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid()

        cls  = output.index_select(2, cls_grid)
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(cls_anchor_dim, nC)

        t1 = time.time()
        grid_x = torch.linspace(0, nW-1, nW).repeat(nB*nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(cls_anchor_dim).to(self.device)
        anchor_w = self.anchors.index_select(1, ix[0]).repeat(nB, nH*nW).view(cls_anchor_dim)
        anchor_h = self.anchors.index_select(1, ix[1]).repeat(nB, nH*nW).view(cls_anchor_dim)

        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        # for build_targets. it works faster on CPU than on GPU
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4)).detach()

        t2 = time.time()
        nGT, nRecall, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), nH, nW)

        cls_mask = (obj_mask == 1)
        tcls = tcls[cls_mask].long().view(-1).to(self.device)
        cls_mask = cls_mask.view(-1, 1).repeat(1,nC).to(self.device)
        cls = cls[cls_mask].view(-1, nC)

        nProposals = int((conf > 0.25).sum())

        tcoord = tcoord.view(4, cls_anchor_dim).to(self.device)
        tconf = tconf.view(cls_anchor_dim).to(self.device)        

        conf_mask = (self.object_scale * obj_mask + self.noobject_scale * noobj_mask).view(cls_anchor_dim).to(self.device)
        obj_mask = obj_mask.view(cls_anchor_dim).to(self.device)
        coord_mask = coord_mask.view(cls_anchor_dim).to(self.device)

        t3 = time.time()
        loss_coord = self.coord_scale * nn.MSELoss(reduction='sum')(coord*coord_mask, tcoord*coord_mask)/nB
        loss_conf = nn.MSELoss(reduction='sum')(conf*conf_mask, tconf*conf_mask)/nB
        loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(cls, tcls)/nB
        loss = loss_coord + loss_conf + loss_cls

        t4 = time.time()
        if False:
            print('-'*30)
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        # print('%d: nGT %3d, nRC %3d, nPP %3d, loss: box %6.3f, conf %6.3f, class %6.3f, total %7.3f'
        #    % (self.seen, nGT, nRecall, nProposals, loss_coord, loss_conf, loss_cls, loss))
        if math.isnan(loss.item()):
            print(conf, tconf)
            sys.exit(0)
        return loss
