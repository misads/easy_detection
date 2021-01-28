from .origin_voc import VOC
from .origin_coco import COCO
from .tile import Tile

datasets = {
    'voc': VOC,  # if --dataset is not specified
    'coco': COCO,
    'tile': Tile,
}

def get_dataset(dataset: str):
    if dataset in datasets:
        return datasets[dataset]
    else:
        raise Exception('No such dataset: "%s", available: {%s}.' % (dataset, '|'.join(datasets.keys())))

