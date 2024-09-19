# encoding=utf-8
from turtle import up
import torch
#torch.multiprocessing.set_sharing_strategy('file_system')  # ulimit -SHn 51200

from dataloader.dataloaders import val_dataloader
from misc_utils import color_print, progress_bar, save_json, get_file_name
from mscv.summary import write_loss, write_image
from options import opt, config

from utils import exception, visualize_boxes
from utils.eval_metrics import eval_detection_voc
from options.helper import init_log
from os.path import join

import cv2
import os
import numpy as np


def eval_mAP(model, 
             dataloader, 
             epoch, 
             writer, 
             logger, 
             log_root, 
             data_name='val'):
    """
    说明:
        验证模型map值, 写入日志, 并可视化(如果指定了--vis参数)

    Args:
        model: network.BaseModel, 模型
        dataloader: torch.utils.data.Dataloader, 验证数据
        epoch: int, 第几个epoch, 用于打印日志
        writer: SummaryWriter, 用于写入tensorboars
        logger: Logger, 用于写入日志
        log_root: str, 保存可视化结果的根目录
        data_name: str, 验证集名称, 有多个验证集时可以指定不同的名称

    Returns:
        None

    """
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    gt_bboxes = []
    gt_labels = []
    gt_difficults = []
    
    if opt.vis and log_root:
        vis_root = join(log_root, 'vis')
        os.makedirs(vis_root, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            progress_bar(i, len(dataloader), 'Eva... ')
            if i > 100:  # debug
                break

            ori_image = sample['ori_image']
            ori_sizes = sample['ori_sizes']

            images = sample['image'] #.to(opt.device)
            gt_bbox = sample['bboxes']
            labels = sample['labels']
            paths = sample['path']

            batch_bboxes, batch_labels, batch_scores = model.forward_test(sample)

            pred_bboxes.extend(batch_bboxes)
            pred_labels.extend(batch_labels)
            pred_scores.extend(batch_scores)

            batch_size = len(gt_bbox)
            for i in range(batch_size):
                gt_bboxes.append(gt_bbox[i].detach().cpu().numpy())
                gt_labels.append(labels[i].int().detach().cpu().numpy())
                gt_difficults.append(np.array([False] * len(gt_bbox[i])))

                if opt.vis:  # 可视化预测结果
                    filename = get_file_name(paths[i])
                    img = ori_image[i]

                    # 缩放bboxes到原图尺寸
                    _, h, w = images[i].shape
                    org_h, org_w = ori_sizes[i]
                    scale_h, scale_w = org_h / h, org_w / w

                    batch_bboxes[i][:, ::2] *= scale_w
                    batch_bboxes[i][:, 1::2] *= scale_h

                    # bgr转rgb, 并可视化bbox
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
                    visualize_boxes(image=img, boxes=batch_bboxes[i],
                                labels=batch_labels[i].astype(np.int32), probs=batch_scores[i], class_labels=config.DATA.CLASS_NAMES)

                    if opt.gt:
                        gt_bbox[i][:, ::2] *= scale_w
                        gt_bbox[i][:, 1::2] *= scale_h
                        for x1, y1, x2, y2 in gt_bbox[i]:
                            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)  # 绿色的是gt
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(join(vis_root, f'{filename}.jpg'), img)
                    # write_image(writer, f'{data_name}/{i}', 'image', img, epoch, 'HWC')


        result = []
        for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            AP = eval_detection_voc(
                pred_bboxes,
                pred_labels,
                pred_scores,
                gt_bboxes,
                gt_labels,
                gt_difficults=None,
                iou_thresh=iou_thresh,
                use_07_metric=False)

            APs = AP['ap']
            mAP = AP['map']
            result.append(mAP)
            if iou_thresh in [0.5, 0.75]:
                logger.info(f'Eva({data_name}) epoch {epoch}, IoU: {iou_thresh}, APs: {str(APs[:10])}, mAP: {mAP}')

            write_loss(writer, f'val/{data_name}', 'mAP', mAP, epoch)

        logger.info(
            f'Eva({data_name}) epoch {epoch}, mean of (AP50-AP95): {sum(result)/len(result)}')



if __name__ == '__main__':
    from options import opt
    from network import get_model
    from mscv.summary import create_summary_writer

    if not opt.load and 'LOAD' not in config.MODEL:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        exception('eval.py: the following arguments are required: --load')

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

    log_root = os.path.join('results', opt.tag, f'epoch_{which_epoch}')
    os.makedirs(log_root, exist_ok=True)

    writer = create_summary_writer(log_root)

    logger = init_log(log_dir=log_root)

    logger.info('===========================================')
    if val_dataloader is not None:
        logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
    logger.info('===========================================')

    eval_mAP(model, val_dataloader, which_epoch, writer, logger, log_root, 'val')

