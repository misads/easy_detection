from .Default.Model import Model as Default
from .Effdet.Model import Model as Effdet
from .Yolo2.Model import Model as Yolo2
from .Faster_RCNN.Model import Model as Faster_RCNN

models = {
    'default': Default,  # if --model is not specified
    'Effdet': Effdet,
    'Yolo2': Yolo2,
    'Faster_RCNN': Faster_RCNN
}

def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

