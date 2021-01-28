import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Tile(object):
    data_format = 'CUSTOM_JSON'  # 自定义的
    voc_root = 'datasets/tile'
    train_split = 'train.txt'
    val_split = 'val.txt' 
    class_names = ['_bkg', 'edge', 'corner', 'whitespot', 'lightblock', 'darkblock', 'aperture']

    img_format = 'jpg'

