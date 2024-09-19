import albumentations as A
from pipelines import transforms as T
from albumentations.pytorch.transforms import ToTensorV2

class FasterRCNN(object):
    def __init__(self, config):
        scale = config.DATA.SCALE or (800, 1333)
        if isinstance(scale, int):
            scale = (scale, scale)
            
        min_scale, max_scale = scale

        # 颜色增强
        if config.DATA.COLOR_AUG:
            train_transform = [
                A.OneOf([
                   A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
                                           val_shift_limit=0.3, p=0.95),
                   A.RandomBrightnessContrast(brightness_limit=0.3,
                                               contrast_limit=0.3, p=0.95),
                ],p=1.0)
            ]
        else:
            train_transform = []

        train_transform += [
            T.FasterRCNNResize(min_scales=min_scale, max_scale=max_scale, p=1.0),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.Normalize(max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0),
        ]

        self.train_transform = A.Compose(  # FRCNN
            train_transform,
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
                T.FasterRCNNResize(min_scales=min_scale, max_scale=max_scale, p=1.0),
                A.Normalize(max_pixel_value=1.0, p=1.0),
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
