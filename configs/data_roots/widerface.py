import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Data(object):
    data_format = 'VOC'
    voc_root = 'datasets/wider_face'
    train_split = 'train.txt'
    val_split = 'val.txt' 
    class_names = ['face']
    img_format = 'jpg'

