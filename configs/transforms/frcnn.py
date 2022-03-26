import albumentations as A
from configs.transforms import custom_transform as C
from albumentations.pytorch.transforms import ToTensorV2

class FRCNN(object):
    def __init__(self, config):
        width = height = short_side = config.DATA.SCALE
        divisor = 32
        self.train_transform = A.Compose(  # FRCNN
            [
                #A.OneOf([
                #    A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
                #                            val_shift_limit=0.3, p=0.95),
                #    A.RandomBrightnessContrast(brightness_limit=0.3,
                #                                contrast_limit=0.3, p=0.95),
                #],p=1.0),

                A.ToGray(p=0.01),
                A.HorizontalFlip(p=0.5),
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

        # FRCNN
        self.val_transform = A.Compose(
            [
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        )
