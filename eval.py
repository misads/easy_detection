# encoding=utf-8

from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim  # deprecated

import dataloader as dl
from options import opt
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im

from PIL import Image
from utils import *

import misc_utils as utils
import pdb


def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):

    save_root = os.path.join(opt.result_dir, opt.tag, str(epoch), data_name)

    utils.try_make_dir(save_root)

    total_loss = 0.
    ct_num = 0
    # print('Start testing ' + tag + '...')
    for i, sample in enumerate(dataloader):
        utils.progress_bar(i, len(dataloader), 'Eva... ')

        loss = model.evaluate(sample)
        total_loss += loss
        
    logger.info(f'Eva({data_name}) epoch {epoch}, total loss: {total_loss}.')

    return f'{total_loss}'



if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)

    opt.which_epoch = model.load(opt.load)

    model.eval()

    log_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
    utils.try_make_dir(log_root)

    writer = create_summary_writer(log_root)

    logger = init_log(training=False)
    evaluate(model, dl.val_dataloader, opt.which_epoch, writer, logger, 'val')

