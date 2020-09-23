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

    total_psnr = 0.0
    total_ssim = 0.0
    ct_num = 0
    # print('Start testing ' + tag + '...')
    for i, data in enumerate(dataloader):
        utils.progress_bar(i, len(dataloader), 'Eva... ')

        input, path = data['input'], data['path']
        img = input.to(device=opt.device)
        recovered = model.inference(img, progress_idx=(i, len(dataloader)))

        if data_name in ['val', 'sots', 'sots_outdoor', 'hsts']:
            label = data['label']
            label = tensor2im(label)

            ct_num += 1
            if i in [0, 20, 40, 60, 80] or data_name=='hsts':
                hazy = tensor2im(input)

                write_image(writer, f'{data_name}/{i}', '1_hazy', hazy, epoch, 'HWC')
                write_image(writer, f'{data_name}/{i}', '2_dehazed', recovered, epoch, 'HWC')
                write_image(writer, f'{data_name}/{i}', '3_label', label, epoch, 'HWC')


            total_psnr += psnr(recovered, label, data_range=255)
            total_ssim += ski_ssim(recovered, label, data_range=255, multichannel=True)

            save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
            Image.fromarray(recovered).save(save_dst)

        elif data_name in ['test', 'real']:
            hazy = tensor2im(input)
            write_image(writer, f'{data_name}/{i}', '1_hazy', hazy, epoch, 'HWC')
            write_image(writer, f'{data_name}/{i}', '2_dehazed', recovered, epoch, 'HWC')

        else:
            raise Exception('Unknown dataset name: %s.' % data_name)

        # 保存结果
        save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
        Image.fromarray(recovered).save(save_dst)

    if data_name in ['val', 'sots', 'sots_outdoor', 'hsts']:
        ave_psnr = total_psnr / float(ct_num)
        ave_ssim = total_ssim / float(ct_num)
        # write_loss(writer, f'val/{data_name}', 'psnr', total_psnr / float(ct_num), epochs)

        logger.info(f'Eva({data_name}) epoch {epoch}, psnr: {ave_psnr}.')
        logger.info(f'Eva({data_name}) epoch {epoch}, ssim: {ave_ssim}.')
        
        return f'{ave_ssim: .3f}'
    else:
        return ''


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

