import torch
import numpy as np

def voc_to_yolo_format(sample):
    """Convert voc to yolo format

    Args:
        sample(dict): {
            'bboxes': bboxes,  # xyxy format, origin size, [N, 4]
            'labels': labels,  # labels, [N]
        }

    Returns:
        {   'yolo_boxes': yolo_boxes,
            'yolo4_boxes': yolo4_boxes,
            'yolo5_boxes': yolo5_boxes
        }

    """

    _, height, width = sample['image'].shape

    yolo_boxes = np.zeros([50, 5])
    yolo4_boxes = np.zeros([50, 5])

    yolo5_boxes = np.zeros([len(sample['bboxes']), 5])
    # sample['bboxes'] = torch.Tensor(sample['bboxes'])

    bboxes = sample['bboxes']
    labels = sample['labels']
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        c_x, c_y = x1 + w / 2, y1 + h / 2
        w, c_x = w / width, c_x / width
        h, c_y = h / height, c_y / height

        if i < yolo5_boxes.shape[0]:
            yolo5_boxes[i, :] = labels[i], c_x, c_y, w, h  # 中心点坐标、宽、高

        if i < 50:
            yolo_boxes[i, :] = labels[i], c_x, c_y, w, h  # 中心点坐标、宽、高
            yolo4_boxes[i, :] = x1, y1, x2, y2, labels[i]

    target = {}
    target['yolo_boxes'] = torch.Tensor(yolo_boxes).view([-1])  # labels, c_x, c_y, w, h (固定50×5)
    target['yolo4_boxes'] = torch.Tensor(yolo4_boxes)
    target['yolo5_boxes'] = torch.Tensor(yolo5_boxes)  #labels, c_x, c_y, w, h (没有固定的bbox数量)

    return target

