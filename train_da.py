# encoding = utf-8
"""
    一个图像复原或分割的Baseline。

    如何添加新的模型：

    ① 复制network目录下的Default文件夹，改成另外一个名字(比如MyNet)。

    ② 在network/__init__.py中import你的Model并且在models = {}中添加它。
        from MyNet.Model import Model as MyNet
        models = {
            'default': Default,
            'MyNet': MyNet,
        }

    ③ 尝试 python train.py --model MyNet 看能否成功运行


    File Structure:
    cv_template
        ├── train.py                :Train and evaluation loop, errors and outputs visualization (Powered by TensorBoard)
        ├── eval.py                 :Evaluation and test (with visualization)
        ├── test.py                 :Test
        │
        ├── clear.py                :Clear cache, be CAREFUL to use it
        │
        ├── run_log.txt             :Record your command logs (except --tag cache)
        │
        ├── network
        │     ├── __init__.py       :Declare all models here so that `--model` can work properly
        │     ├── Default
        │     │      ├── Model.py   :Define default model, losses and parameter updating procedure
        │     │      └── FFA.py
        │     └── MyNet
        │            ├── Model.py   :Define your model, losses and parameter updating procedure
        │            └── mynet.py
        ├── options
        │     └── options.py        :Define options
        │
        │
        ├── dataloader/             :Define Dataloaders
        │     ├── __init__.py       :imports all dataloaders in dataloaders.py
        │     ├── dataloaders.py    :Define all dataloaders here
        │     └── my_dataset.py     :Custom Dataset
        │
        ├── checkpoints/<tag>       :Trained checkpoints
        ├── logs/<tag>              :Logs and TensorBoard event files
        └── results/<tag>           :Test results


    Datasets:

        datasets
           ├── train
           │     ├── 00001
           │     ├── 00002
           │     └── .....
           ├──  val
           │     ├── 00001
           │     ├── 00002
           │     └── .....
           ├── train.txt
           └── val.txt

    Usage:

    #### Train

        python train.py --tag train_1 --epochs 500 -b 8 --gpu 1

    #### Resume Training

        python train.py --load checkpoints/train_1/500_checkpoint.pt --resume

    #### Evaluation

        python eval.py --tag eval_1 --model MyNet --load checkpoints/train_1/500_checkpoint.pt

    #### Test

        python test.py --tag test_1

    #### Clear

        python clear.py [my_tag]  # (DO NOT use this command unless you know what you are doing.)


    License: MIT

"""

import os
import pdb
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

import dataloader as dl
from network import get_model
from eval import evaluate
from eval_yolo import eval_yolo

from options import opt

from utils import init_log
from mscv.summary import create_summary_writer, write_meters_loss, write_image
from mscv.image import tensor2im
# from utils.send_sms import send_notification

import misc_utils as utils

######################
#       Paths
######################
save_root = os.path.join(opt.checkpoint_dir, opt.tag)
log_root = os.path.join(opt.log_dir, opt.tag)

utils.try_make_dir(save_root)
utils.try_make_dir(log_root)


######################
#      DataLoaders
######################
train_dataloader = dl.train_dataloader
val_dataloader = dl.val_dataloader

src_data_loader, tgt_data_loader = dl.src_data_loader, dl.tgt_data_loader
len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

# init log
logger = init_log(training=True)

######################
#     Init model
######################
Model = get_model(opt.model)
model = Model(opt)

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

# Start training


print('Start training...')
start_step = (start_epoch - 1) * len_dataloader
global_step = start_step
total_steps = opt.epochs * len_dataloader
start = time.time()

#####################
#   定义scheduler
#####################

scheduler = model.scheduler

######################
#    Summary_writer
######################
writer = create_summary_writer(log_root)

start_time = time.time()
######################
#     Train loop
######################
# for iteration, data in enumerate(train_dataloader):
#     rate = (iteration) / (time.time() - start)
#     remaining = (len(train_dataloader) - iteration) / rate if rate else 9999999
#     utils.progress_bar(iteration, len(train_dataloader), 'Step:', f'ETA: {utils.format_time(remaining)}')


"""
import cv2
for dataname, dataloader in [('src', dl.src_data_loader), ('tgt', dl.tgt_data_loader)]:
    for i, data in enumerate(dataloader):
        if i >= 9:
            break

        input = data[0]
        bboxes = data[1][0]
        bboxes = bboxes.view([-1, 5])

        img = tensor2im(input)

        for line in bboxes:
            label, cx, cy, lx, ly = line
            x2 = int((cx+lx/2) * 416)
            y2 = int((cy+ly/2) * 416)
            x1 = int((cx-lx/2) * 416)
            y1 = int((cy-ly/2) * 416)

            img = img.copy()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        write_image(writer, dataname, f'input/{i}', img, 1, 'HWC')

import ipdb; ipdb.set_trace()
"""

MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()
# model.eval()

# eval_yolo(model.DA.Yolo, dl.src_val_loader, 0, writer, logger, 'src')
# eval_yolo(model.DA.Yolo, dl.tgt_val_loader, 0, writer, logger, 'tgt')

# model.train()


try:
    eval_result = ''

    for epoch in range(start_epoch, opt.epochs + 1):
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for iteration, ((images_src, class_src), (images_tgt, _)) in data_zip:
            global_step += 1
            # model.adjust_learning_rate(global_step)

            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            if opt.debug and iteration > 10:
                break

            # img, label = data['input'], data['label']  # ['label'], data['image']  #
            # img, label = data

            img_src_var = Variable(images_src, requires_grad=False).to(device=opt.device)
            img_tgt_var = Variable(images_tgt, requires_grad=False).to(device=opt.device)
            label_var = Variable(class_src, requires_grad=False).to(device=opt.device)

            ##############################
            #       Update parameters
            ##############################
            p = global_step / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = .1

            update = model.update(img_src_var, label_var, img_tgt_var, alpha)
            # recovered = update.get('recovered')

            pre_msg = 'Epoch:%d' % epoch

            msg = f'lr:{round(scheduler.get_lr()[0], 6) : .6f}, alpha: {alpha:.6f}, (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            utils.progress_bar(iteration, len_dataloader, pre_msg, msg)
            # print(pre_msg, msg)

            if global_step % 1000 == 0:
                write_meters_loss(writer, 'train', model.avg_meters, global_step)
                # hazy = tensor2im(img)
                # dehazed = tensor2im(recovered)
                # gt = tensor2im(label)
                # write_image(writer, 'train', '1_hazy', hazy, global_step, 'HWC')
                # write_image(writer, 'train', '2_dehazed', dehazed, global_step, 'HWC')
                # write_image(writer, 'train', '3_label', gt, global_step, 'HWC')

        logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            model.save(epoch)

        ####################
        #     Validation
        ####################
        if epoch % opt.eval_freq == 0:

            model.eval()

            cnt = 0
            meanmse = 0
            meanl1 = 0
            for j, data in enumerate(dl.paired_loader):
                image, gt, _, _ = data
                image = image.to(opt.device)
                gt = gt.to(opt.device)
                utils.progress_bar(j, len(dl.paired_loader), 'Eva... ')
                with torch.no_grad():
                    hazy_detection_output, hazy_domain_outputs = model.DA(image, alpha=0.1)
                    clear_detection_output, clear_domain_outputs = model.DA(gt, alpha=0.1)
                    hazy_feature = model.DA.feature(image)
                    hazy_feature = hazy_feature[20]
                    clear_feature =  model.DA.feature(gt)
                    clear_feature = clear_feature[20]

                    mseloss = MSE(hazy_feature, clear_feature)
                    l1loss = L1(hazy_feature, clear_feature)
                    meanmse += mseloss
                    meanl1 += l1loss
                    cnt += image.size(0)

                if j < 48:
                    vis_hazy = tensor2im(hazy_domain_outputs[0])
                    vis_gt = tensor2im(clear_domain_outputs[0])

                    write_image(writer, f'paired/{j}', '1_hazy', vis_hazy, epoch, 'HWC')
                    write_image(writer, f'paired/{j}', '2_clear', vis_gt, epoch, 'HWC')

            ones = np.ones([13, 13, 1])
            write_image(writer, f'paired/label', '1_hazy', ones, epoch, 'HWC')
            zeros = np.zeros([13, 13, 1])
            write_image(writer, f'paired/label', '2_clear', zeros, epoch, 'HWC')
            

            meanmse = meanmse / cnt * 100
            meanl1 = meanl1 / cnt * 1000
            
            logger.info(f'Eva(paired) epoch {epoch}, MSE: {meanmse}, L1: {meanl1}')
                    
            eval_yolo(model.DA.Yolo, dl.src_val_loader, epoch, writer, logger, 'src')
            eval_yolo(model.DA.Yolo, dl.tgt_val_loader, epoch, writer, logger, 'tgt')
            # eval_result = evaluate(model, val_dataloader, epoch, writer, logger)
            # sots_result = evaluate(model, dl.sots_dataloader, epoch, writer, logger, 'sots')
            # _ = evaluate(model, dl.sots_outdoor_dataloader, epoch, writer, logger, 'sots_outdoor')
            # _ = evaluate(model, dl.hsts_dataloader, epoch, writer, logger, 'hsts')
            # _ = evaluate(model, dl.real_dataloader, epoch, writer, logger, 'real')

            model.train()

        if scheduler is not None:
            scheduler.step()

    # send_notification([opt.tag[:12], '', '', eval_result])

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Accuracy:' + eval_result + '\n')

except Exception as e:

    # if not opt.debug:  # debug模式不会发短信 12是短信模板字数限制
    #     send_notification([opt.tag[:12], str(e)[:12]], template='error')

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    # print(e)
    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的trace back信息
