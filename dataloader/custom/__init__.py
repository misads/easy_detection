from .real_voc import VOC
from .real_coco import COCO
from .apollo import Apollo
# from .apollo_hazy import Apollo_hazy
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
    # 'apollo_hazy': Apollo_hazy
}

def get_dataset(dataset: str):
    if dataset in datasets:
        return datasets[dataset]
    else:
        raise Exception('No such dataset: "%s", available: {%s}.' % (dataset, '|'.join(datasets.keys())))

