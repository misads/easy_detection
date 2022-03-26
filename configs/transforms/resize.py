import albumentations as A
from configs.transforms import custom_transform as C
from albumentations.pytorch.transforms import ToTensorV2

class Resize(object):
    def __init__(self, config):
        width, height = config.DATA.SCALE  # 416

        self.train_transform = A.Compose(
            [
                A.Resize(height=height, width=width, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            ),
        )

        self.val_transform = train_transform

