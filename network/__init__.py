from .Default.Model import Model as Default
from .Effdet.Model import Model as Effdet
from .Yolo.Model import Model as Yolo2
from .Yolo.Model import Model as Yolo3
from .SSD.Model import Model as SSD300
from .SSD.Model import Model as SSD512
from .RetinaNet.Model import Model as RetinaNet
from .Faster_RCNN.Model import Model as Faster_RCNN
from .YoloV5.Model import Model as Yolo5

models = {
    'default': Default,  # if --model is not specified
    'Effdet': Effdet,
    'Yolo2': Yolo2,
    'Yolo3': Yolo3,
    'Faster_RCNN': Faster_RCNN,
    'Yolo5': Yolo5,
    'SSD300': SSD300,
    'SSD512': SSD512,
    'RetinaNet':RetinaNet
}

def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

