import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Data(object):
    data_format = 'VOC'
    voc_root = 'datasets/cityscapes'
    train_split = 'train.txt'
    val_split = 'val.txt' 
    class_names = ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']
    img_format = 'png'
