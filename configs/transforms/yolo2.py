import albumentations as A
from configs.transforms import custom_transform as C
from albumentations.pytorch.transforms import ToTensorV2
from options import opt

class Yolo2(object):
    def __init__(self, config):
        width, height = config.DATA.SCALE

        self.train_transform = A.Compose(  # Yolo
            [
                # A.RandomSizedCrop(min_max_height=(800, 1024), height=1024, width=1024, p=0.5),
                # A.RandomScale(scale_limit=0.3, p=1.0),  # 这个有问题
                C.RandomResize(scale_limit=0.3, p=1.0),  # 调节长宽比 [1/1.3, 1.3]
                A.OneOf([
                    A.Sequential(
                        [
                            A.SmallestMaxSize(min(height, width), p=1.0),
                            A.RandomCrop(height, width, p=1.0)  # 先resize到短边544，再crop成544×544
                        ],
                    p=0.4),
                    A.LongestMaxSize(max(height, width), p=0.6),  #  resize到长边544
                ], p=1.0),

                # A.LongestMaxSize(max(height, width), p=1.0),

                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.4, sat_shift_limit=0.4,
                                            val_shift_limit=0.4, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.3,
                                                contrast_limit=0.3, p=0.9),
                ],p=0.9),
                # A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=(0.5,0.5,0.5), p=1.0),
                C.RandomPad(min_height=height, min_width=width, border_mode=0, value=(0.5,0.5,0.5), p=1.0),
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


        divisor = 32
        self.val_transform = A.Compose(  # Yolo
            [
                A.LongestMaxSize(width, p=1.0),
                A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=(0.5,0.5,0.5), p=1.0),
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
