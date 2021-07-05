import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Data(object):
    data_format = 'VOC'
    voc_root = 'datasets/voc'
    train_split = 'train.txt'
    val_split = 'test.txt' 
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    img_format = 'jpg'

