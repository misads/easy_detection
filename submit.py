# encoding=utf-8
import torch
torch.multiprocessing.set_sharing_strategy('file_system')  # ulimit -SHn 51200

#from skimage.measure import compare_psnr as psnr
#from skimage.measure import compare_ssim as ski_ssim  # deprecated

from dataloader.dataloaders import val_dataloader, test_dataloader
from options import opt
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im

from PIL import Image
from utils import *
from utils.vis import visualize_boxes

import misc_utils as utils
import pdb
import cv2
import os

keep_thresh = 0.0

def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):
    # 每个模型的evaluate方式不同
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    names = []

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            name = os.path.basename(sample['path'][0])
            names.append(name)
            utils.progress_bar(i, len(dataloader), 'Eva... ')
            image = sample['image'].to(opt.device)
            paths = sample['path']

            batch_bboxes, batch_labels, batch_scores = model.forward_test(image)
            pred_bboxes.extend(batch_bboxes)
            pred_labels.extend(batch_labels)
            pred_scores.extend(batch_scores)

            if opt.vis:  # 可视化预测结果
                img = tensor2im(image).copy()

                num = len(batch_scores[0])
                visualize_boxes(image=img, boxes=batch_bboxes[0],
                            labels=batch_labels[0].astype(np.int32), probs=batch_scores[0], class_labels=opt.class_names)

                write_image(writer, f'{data_name}/{i}', 'image', img, epoch, 'HWC')

    submit = []
    for i in range(len(pred_labels)):
        keep = pred_scores[i] > keep_thresh
        pred_bboxes[i] = pred_bboxes[i][keep]
        pred_labels[i] = pred_labels[i][keep]
        pred_scores[i] = pred_scores[i][keep]

        for j in range(len(pred_scores[i])):
            line = {
                "name": names[i],
                "category": int(pred_labels[i][j]),
                "bbox": pred_bboxes[i][j].tolist(),
                "score": float(pred_scores[i][j])
            }
            submit.append(line)

    try:
        suffix = '' if opt.ms == 1. else '_' + str(opt.ms)
        with open(f'result{suffix}.json', 'w', encoding='utf-8') as f:
            json.dump(submit, f, indent=4, ensure_ascii=False)
    except:
        pass

    # import ipdb
    # ipdb.set_trace() 
   


if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load and not opt.weights:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)

    if opt.load:
        opt.which_epoch = model.load(opt.load)
    else:
        opt.which_epoch = 0

    model.eval()

    log_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
    utils.try_make_dir(log_root)

    writer = create_summary_writer(log_root)

    logger = init_log(training=False)

    logger.info('===========================================')
    if test_dataloader is not None:
        logger.info('test_trasforms: ' +str(test_dataloader.dataset.transforms))
    logger.info('===========================================')

    evaluate(model, test_dataloader, opt.which_epoch, writer, logger, 'test')

