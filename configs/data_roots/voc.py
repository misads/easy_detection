import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class VOC(object):
    data_format = 'VOC'
    voc_root = 'datasets/voc'
    train_split = 'train.txt'
    val_split = 'test.txt' 
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # class_names = [
    #     'person', 'bicycle', 'car', 'motorbike', 'aeroplane',
    #     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    #     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    #     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    #     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    #     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    #     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    #     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    #     'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
    #     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    #     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    # ]

    img_format = 'jpg'

