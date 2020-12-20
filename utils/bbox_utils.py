from typing import Iterable

import torch
import numpy as np

def keep(condition, arr):
    return [a[condition] for a in arr]


def to_numpy(source, dtype=None):
    if isinstance(source, torch.Tensor):
        ans = source.detach().cpu().numpy()
    elif isinstance(source, Iterable):
        ans = np.array(source)
    else:
        ans = np.array([source])

    if dtype is not None:
        ans = ans.astype(dtype)

    return ans


def xywh_to_xyxy(boxes, width=1.0, height=1.0):
    """
    Convert bbox from xywh format to xyxy format.
    Parameters
    ----------
    boxes : Tensor[N, 4])
        They are expected to be in (x, y, w, h) format
    width : float
        DeNorm bbox from range [0, 1] to image size  
    height : float
        DeNorm bbox from range [0, 1] to image size  


    Returns
    -------
    boxes: Tensor[N, 4])
        in (x, y, x, y) format
    

    """
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    boxes = torch.clamp(boxes, min=0, max=1)
    
    boxes[:, 0] *= width
    boxes[:, 2] *= width
    boxes[:, 1] *= height
    boxes[:, 3] *= height

    return boxes

