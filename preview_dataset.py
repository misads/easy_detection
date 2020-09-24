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
import misc_utils as utils
from dataloader.image_list import ListTrainValDataset, ListTestDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader import yolo_dataset

from dataloader import effdet_dataset 
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
from dataloader import voc
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

train_transform = train_transform = A.Compose(
    [
        A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                    val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                        contrast_limit=0.2, p=0.9),
        ],p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=512, width=512, p=1),
        A.Cutout(num_holes=5, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        ToTensorV2(p=1.0),
    ], 
    p=1.0, 
    bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0, 
        min_visibility=0,
        label_fields=['labels']
    ),
)

voc_dataset = voc.VOCTrainValDataset('/home/raid/public/datasets/wheat_detection', 
        ['wheat'],
        split='train.txt',
        transforms=train_transform)

def collate_fn(batch):
    target = {}
    target['image'] = torch.stack([sample['image'] for sample in batch])
    target['bboxes'] = [sample['bboxes'] for sample in batch]
    target['labels'] = [sample['labels'] for sample in batch]
    target['path'] = [sample['path'] for sample in batch]
    target['yolo_boxes'] = torch.stack([sample['yolo_boxes'] for sample in batch])
    return target

voc_dataloader = torch.utils.data.DataLoader(voc_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=opt.batch_size,
    num_workers=0,
    drop_last=True)
    
import ipdb; ipdb.set_trace()
for i, sample in enumerate(voc_dataloader):
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
