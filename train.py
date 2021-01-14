# encoding = utf-8
import os
import pdb
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

from dataloader.dataloaders import train_dataloader, val_dataloader

from network import get_model
from eval import evaluate

from options import opt
from scheduler import schedulers

from utils import init_log, seed_everything, load_meta, save_meta
from mscv.summary import create_summary_writer, write_meters_loss, write_image
from mscv.image import tensor2im
# from utils.send_sms import send_notification

import misc_utils as utils
import random
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

# 初始化
with torch.no_grad():
    # 设置随机种子
    if opt.seed is not None:
        seed_everything(opt.seed)
    
    # 初始化路径
    save_root = os.path.join(opt.checkpoint_dir, opt.tag)
    log_root = os.path.join(opt.log_dir, opt.tag)

    utils.try_make_dir(save_root)
    utils.try_make_dir(log_root)

    # dataloader
    train_dataloader = train_dataloader
    val_dataloader = val_dataloader

    # 初始化日志
    logger = init_log(training=True)

    # 初始化训练的meta信息
    meta = load_meta(new=True)
    save_meta(meta)

    # 初始化模型
    Model = get_model(opt.model)
    model = Model(opt, logger)

    # 暂时还不支持多GPU
    # if len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model = model.to(device=opt.device)

    if opt.load:
        load_epoch = model.load(opt.load)
        start_epoch = load_epoch + 1 if opt.resume else 1
    else:
        start_epoch = 1

    model.train()

    # 开始训练
    print('Start training...')
    start_step = (start_epoch - 1) * len(train_dataloader)
    global_step = start_step
    total_steps = opt.epochs * len(train_dataloader)
    start = time.time()

    # 定义scheduler
    scheduler = model.scheduler

    # tensorboard日志
    writer = create_summary_writer(log_root)

    start_time = time.time()

    # 在日志记录transforms
    logger.info('train_trasforms: ' +str(train_dataloader.dataset.transforms))
    logger.info('===========================================')
    if val_dataloader is not None:
        logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
    logger.info('===========================================')

    # 在日志记录scheduler
    if opt.scheduler in schedulers:
        logger.info('scheduler: (Lambda scheduler)\n' + str(schedulers[opt.scheduler]))
        logger.info('===========================================')

# 训练循环
try:
    eval_result = ''

    for epoch in range(start_epoch, opt.epochs + 1):
        for iteration, sample in enumerate(train_dataloader):
            global_step += 1

            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            # --debug模式下只训练10个batch
            if opt.debug and iteration > 10:
                break

            sample['global_step'] = global_step
     
            #  更新网络参数
            updated = model.update(sample)
            predicted = updated.get('predicted')

            pre_msg = 'Epoch:%d' % epoch

            # 显示进度条
            msg = f'lr:{round(scheduler.get_lr()[0], 6) : .6f} (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            utils.progress_bar(iteration, len(train_dataloader), pre_msg, msg)
            # print(pre_msg, msg)

            if global_step % 1000 == 0:  # 每1000个step将loss写到tensorboard
                write_meters_loss(writer, 'train', model.avg_meters, global_step)

        # 记录训练日志
        logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            model.save(epoch)

  
        # 训练时验证
        if not opt.no_eval and epoch % opt.eval_freq == 0:

            model.eval()
            evaluate(model, val_dataloader, epoch, writer, logger, data_name='val')
            model.train()

        if scheduler is not None:
            scheduler.step()

    # 保存结束信息
    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Accuracy:' + eval_result + '\n')

    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)

except Exception as e:

    # if not opt.debug:  # debug模式不会发短信 12是短信模板字数限制
    #     send_notification([opt.tag[:12], str(e)[:12]], template='error')

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)
    # print(e)
    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的trace back信息

except:  # 其他异常，如键盘中断等
    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)