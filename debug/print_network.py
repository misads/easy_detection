# encoding: utf-8
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from options import opt, config
from network import get_model
from misc_utils import color_print

if __name__ == '__main__':
    opt.debug = False
    config.DATA.NUM_CLASSESS = 80

    Model = get_model(config.MODEL.NAME)
    model = Model(config)

    detector = model.detector
    for key in detector._modules.keys():
        if key not in ['transform']:  # 'backbone', 
            color_print(key, 3)
            print(detector._modules[key])


