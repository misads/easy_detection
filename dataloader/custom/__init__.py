from .origin_voc import VOC
from .origin_coco import COCO
from .apollo import Apollo
from .cityscapes import Cityscapes
from .wheat import Wheat
from .widerface import Wider_face
from .rtts import Rtts

datasets = {
    'voc': VOC,  # if --dataset is not specified
    'coco': COCO,
    'apollo': Apollo,  
    'cityscapes': Cityscapes,
    'wheat': Wheat,
    'widerface': Wider_face,
    'rtts': Rtts,
}

def get_dataset(dataset: str):
    if dataset in datasets:
        return datasets[dataset]
    else:
        raise Exception('No such dataset: "%s", available: {%s}.' % (dataset, '|'.join(datasets.keys())))

