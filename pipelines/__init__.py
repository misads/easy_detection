from .no_transform import No_Transform
from .resize import Resize
from .faster_rcnn import FasterRCNN
from .yolo2 import Yolo2
from .ssd import SSD

available_transforms = {
    'yolo2': Yolo2,
    'yolo3': Yolo2,
    'faster_rcnn': FasterRCNN,
    'ssd300': SSD,
    'ssd512': SSD,
    'none': No_Transform,
    'resize': Resize,
    'None': No_Transform,
}


def get_transform(transform_name: str):
    if transform_name in available_transforms:
        return available_transforms[transform_name]
    else:
        raise Exception('No such transform: "%s", available: {%s}.' % (transform_name, '|'.join(available_transforms.keys())))

