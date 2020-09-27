from .Default.Model import Model as Default
from .Effdet.Model import Model as Effdet
from .Yolo.Model import Model as Yolo2
from .Yolo.Model import Model as Yolo3

from .Faster_RCNN.Model import Model as Faster_RCNN
from .YoloV5.Model import Model as Yolo5

models = {
    'default': Default,  # if --model is not specified
    'Effdet': Effdet,
    'Yolo2': Yolo2,
    'Yolo3': Yolo3,
    'Faster_RCNN': Faster_RCNN,
    'Yolo5': Yolo5
}

def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

