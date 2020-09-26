# encoding: utf-8
# from dataloader.paired_dataset import pairedDataset
#

# from torchvision import datasets, transforms
#
#
# paired_list = './datasets/apollo/apollo_val.txt'  # 这里要填一下
#
# dataset = pairedDataset(paired_list, shape=(416, 416),  # 这里不是listDataset
#             shuffle=False,
#             transform=transforms.Compose([
#                 transforms.ToTensor(),
#             ]))
#
# writer = create_summary_writer('logs/preview_dataset')
#
# for i in range(10):
#     data = dataset[i]
#
#     image, gt, _, _ = data
#     # import ipdb; ipdb.set_trace()
#
#     write_image(writer, f'preview/{i}', 'input', image, 0, 'CHW')
#     write_image(writer, f'preview/{i}', 'label', gt, 0, 'CHW')
#
# utils.color_print('dataset预览已保存到logs/preview_dataset', 3)
from mscv.summary import create_summary_writer, write_image
from dataloader import voc
# from dataloader import paired_dataset

import cv2
from options import opt
from mscv.image import tensor2im
import torch
import ipdb

"""
source domain 是clear的
"""
writer = create_summary_writer('logs/preview_dataset')
from dataloader.dataloaders import train_dataloader

import ipdb; ipdb.set_trace()
for i, sample in enumerate(train_dataloader):
    pass

exit(1)

#     image = sample['image'][0].detach().cpu().numpy().transpose([1,2,0])
#     image = image.copy()
#     bboxes = sample['bboxes'][0]
#     for y1, x1, y2, x2 in bboxes:
#         x1 = int(round(x1.item()))
#         y1 = int(round(y1.item()))
#         x2 = int(round(x2.item()))
#         y2 = int(round(y2.item()))
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 1., 0), 2)

#     write_image(writer, f'preview/{i}', 'image', image, 0, 'HWC')

for i, sample in enumerate(effdet_dataset.eff_train_loader):
    if i > 30:
        break

    images, bboxes, image_id = sample

    image = images[0].detach().cpu().numpy().transpose([1,2,0])
    image = image.copy()
    bboxes = bboxes[0]['boxes']
    for y1, x1, y2, x2 in bboxes:
        x1 = int(round(x1.item()))
        y1 = int(round(y1.item()))
        x2 = int(round(x2.item()))
        y2 = int(round(y2.item()))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 1., 0), 2)

    write_image(writer, f'preview/{i}', 'image', image, 0, 'HWC')

writer.flush()
