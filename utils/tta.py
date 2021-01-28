import torch
import torch.nn as nn
import numpy as np
import ipdb
from torchvision.ops import nms
from mmdet.ops.nms import batched_nms

nms_type = 'soft_nms'
nms_type = 'nms'
nms_thresh = 0.1

def tta_wrapper(nw: int, nh: int, patch_w: int, patch_h: int):
    def decorator(fn):
        def handle(*args, **kwargs):
            image = args[1]  # 0是self
            b, c, h, w = image.shape
            assert b == 1, 'batch size must be 1'

            overlap_w = (nw * patch_w - w) // (nw - 1) 
            overlap_h = (nh * patch_h - h) // (nh - 1)
            startw = []
            starth = []
            for i in range(nw-1):
                if i == 0:
                    startw.append(0)
                else:
                    startw.append(startw[-1]+patch_w-overlap_w)
            startw.append(w-patch_w)

            for i in range(nh-1):
                if i == 0:
                    starth.append(0)
                else:
                    starth.append(starth[-1]+patch_h - overlap_h)
            starth.append(h-patch_h)
            batch_bboxes = None
            batch_labels = None
            batch_scores = None
            for i in range(nh):
                for j in range(nw):
                    try:
                        patch = image[:, :, starth[i]:starth[i]+patch_h, startw[j]:startw[j]+patch_w]
                    except:
                        ipdb.set_trace()

                    # patch_bboxes, patch_labels, patch_scores = model.forward_test(patch)
                    patch_bboxes, patch_labels, patch_scores = fn(args[0], patch)

                    patch_bboxes = patch_bboxes[0]
                    patch_labels = patch_labels[0]
                    patch_scores = patch_scores[0]

                    for k in range(len(patch_bboxes)):
                        patch_bboxes[k][0] += startw[j]
                        patch_bboxes[k][1] += starth[i]
                        patch_bboxes[k][2] += startw[j]
                        patch_bboxes[k][3] += starth[i]
                    if batch_bboxes is not None:
                        batch_bboxes = np.concatenate((batch_bboxes, patch_bboxes), axis=0)
                        batch_labels = np.concatenate((batch_labels, patch_labels), axis=0)
                        batch_scores = np.concatenate((batch_scores, patch_scores), axis=0)
                    else:
                        batch_bboxes, batch_labels, batch_scores = patch_bboxes, patch_labels, patch_scores

            if len(batch_labels) > 0:
                nms_cfg = {'type': nms_type, 'iou_thr': nms_thresh}
                dets, keep = batched_nms(torch.Tensor(batch_bboxes).cuda(), 
                                        torch.Tensor(batch_scores).cuda(), 
                                        torch.Tensor(batch_labels).cuda(), nms_cfg)
                # import ipdb
                # ipdb.set_trace()
                # keep = nms(torch.Tensor(batch_bboxes).cuda(), torch.Tensor(batch_scores).cuda(), nms_thresh)
                keep = keep.detach().cpu().numpy()
                batch_bboxes, batch_labels, batch_scores = batch_bboxes[keep], batch_labels[keep], batch_scores[keep]

            return np.array([batch_bboxes]), np.array([batch_labels]), np.array([batch_scores])
            # result = fn(*args, **kwargs)
            # return result

        return handle

    return decorator

