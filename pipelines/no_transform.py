import albumentations as A
from pipelines import transforms as T
from albumentations.pytorch.transforms import ToTensorV2

class No_Transform(object):
    width = height = 1000

    train_transform = A.Compose(  # FRCNN
        [
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

    val_transform = train_transform

