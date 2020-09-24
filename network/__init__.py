from .Default.Model import Model as Default
# from .PureV2_Yolo.Model import Model as PureV2_Yolo
# from .DA.Model import Model as DA
# from .DA_Resnet.Model import Model as DA_Resnet
from .Effdet.Model import Model as Effdet
from .Yolo2.Model import Model as Yolo2

models = {
    'default': Default,  # if --model is not specified
    # 'PureV2_Yolo': PureV2_Yolo,
    # 'DA': DA,
    # 'DA_Resnet': DA_Resnet,
    'Effdet': Effdet,
    'Yolo2': Yolo2
}

def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

