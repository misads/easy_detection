from .no_transform import No_Transform
from .resize import Resize
from .frcnn import FRCNN
from .yolo2 import Yolo2
from .ssd import SSD

transforms = {
    'yolo2': Yolo2,
    'yolo3': Yolo2,
    'faster_rcnn': FRCNN,
    'frcnn': FRCNN,
    'ssd300': SSD,
    'ssd512': SSD,
    'none': No_Transform,
    'resize': Resize,
    None: No_Transform,
}


def get_transform(transform: str):
    if transform in transforms:
        return transforms[transform]
    else:
        raise Exception('No such transform: "%s", available: {%s}.' % (transform, '|'.join(transforms.keys())))

