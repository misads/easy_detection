import albumentations as A
from configs.transforms import custom_transform as C
from albumentations.pytorch.transforms import ToTensorV2
from options import opt

class FRCNN(object):
    width = height = short_side = opt.scale if opt.scale else 600

    divisor = 32
    train_transform = A.Compose(  # FRCNN
        [
            A.SmallestMaxSize(short_side, p=1.0),  # resize到短边600
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),

            # A.RandomCrop(height=height, width=width, p=1.0),  # 600×600
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3,
                                        val_shift_limit=0.3, p=0.95),
                A.RandomBrightnessContrast(brightness_limit=0.3,
                                            contrast_limit=0.3, p=0.95),
            ],p=1.0),

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
    val_transform = A.Compose(
        [
            A.SmallestMaxSize(short_side, p=1.0),  # resize到短边600
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),
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
