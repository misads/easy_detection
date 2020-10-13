import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Wider_face(object):
    data_format = 'VOC'
    voc_root = 'datasets/wider_face'
    train_split = 'train.txt'
    val_split = 'val.txt' 
    class_names = ['face']
    img_format = 'jpg'

    width = 512
    height = 512

    train_transform = A.Compose(
        [
            A.Resize(height=1024, width=1024, p=1),  # 896
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                        val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                            contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.Resize(height=height, width=width, p=1),
            # A.Cutout(num_holes=5, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
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

    val_transform = A.Compose(
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
        )
    )
