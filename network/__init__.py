from .Effdet.Model import Model as Effdet
from .YoloV2V3.Model import Model as Yolo2
from .YoloV2V3.Model import Model as Yolo3
from .SSD.Model import Model as SSD300
from .SSD.Model import Model as SSD512
from .RetinaNet.Model import Model as RetinaNet
from .Faster_RCNN.Model import Model as Faster_RCNN
from .Cascade_RCNN.Model import Model as Cascade_RCNN
from .YoloV5.Model import Model as Yolo5
from .YoloV4.Model import Model as Yolo4

models = {
    'Effdet': Effdet,
    'Yolo2': Yolo2,
    'Yolo3': Yolo3,
    'Faster_RCNN': Faster_RCNN,
    'Cascade_RCNN': Cascade_RCNN,
    'FRCNN': Faster_RCNN,
    'Yolo5': Yolo5,
    'SSD300': SSD300,
    'SSD512': SSD512,
    'RetinaNet':RetinaNet,
    'Yolo4':Yolo4
}

def get_model(model: str):
    if model is None:
        raise AttributeError('--model MUST be specified now, available: {%s}.' % ('|'.join(models.keys())))

    if model in models:
        return models[model]
    else:
        raise AttributeError('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

