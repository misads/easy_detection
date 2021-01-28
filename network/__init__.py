from .Faster_RCNN.Model import Model as Faster_RCNN
from .YoloV5.Model import Model as Yolo5

models = {
    'Faster_RCNN': Faster_RCNN,
    'FRCNN': Faster_RCNN,
    'Yolo5': Yolo5,
}

def get_model(model: str):
    if model is None:
        raise AttributeError('--model MUST be specified now, available: {%s}.' % ('|'.join(models.keys())))

    if model in models:
        return models[model]
    else:
        raise AttributeError('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

