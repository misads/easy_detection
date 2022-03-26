# encoding=utf-8
import torch
#torch.multiprocessing.set_sharing_strategy('file_system')  # ulimit -SHn 51200

from dataloader.dataloaders import val_dataloader
from options import opt, config
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im

from PIL import Image
from utils import *
from options.helper import init_log

import misc_utils as utils
import pdb


def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):
    # 每个模型的evaluate方式不同
    loss = model.evaluate(dataloader, epoch, writer, logger, data_name)
    return f'{loss}'


if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load and 'LOAD' not in config.MODEL:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(config.MODEL.NAME)
    model = Model(config)
    model = model.to(device=opt.device)

    if opt.load:
        which_epoch = model.load(opt.load)
    elif 'LOAD' in config.MODEL:
        which_epoch = model.load(config.MODEL.LOAD)
    else:
        which_epoch = 0

    model.eval()

    log_root = os.path.join('results', opt.tag, str(which_epoch))
    utils.try_make_dir(log_root)

    writer = create_summary_writer(log_root)

    logger = init_log(training=False)

    logger.info('===========================================')
    if val_dataloader is not None:
        logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
    logger.info('===========================================')

    evaluate(model, val_dataloader, which_epoch, writer, logger, 'val')

