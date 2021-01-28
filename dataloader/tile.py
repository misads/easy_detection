# encoding=utf-8
import ipdb

import torch
import torchvision.transforms.functional as F
import os
from PIL import Image
from torchvision import transforms
from options import opt
import torch.utils.data.dataset as dataset
import albumentations as A

import misc_utils as utils
import random
import pickle
import numpy as np
import cv2

from collections import defaultdict
from dataloader.additional import voc_to_yolo_format

limit = lambda x, minimum, maximum: min(max(minimum, x), maximum)


class TILETrainValDataset(dataset.Dataset):
    """TILE Dataset for training and validation.

    Args:
        voc_root(str): root dir to dataset
        class_names(list(str)): class names
        split(str): .txt file in ImageSets/Main
        transforms(albumentations.transform): required, input images and bboxes will be applied simultaneously
        max_size(int): maximum data returned
    """

    def __init__(self, voc_root, class_names, split='train.txt', transforms=None, max_size=None):
        voc_root = os.path.join(voc_root, 'tile_round1_train_20201231')

        utils.color_print(f'Use dataset: {voc_root}, split: {split[:-4]}', 3)

        im_list = f'{voc_root}/{split}'
        image_root = f'{voc_root}/train_imgs'
        anno_file = f'{voc_root}/train_annos.json'

        self.image_paths = set()
        self.bboxes = defaultdict(list)
        self.labels = defaultdict(list)

        counter = defaultdict(int)
        tot_bbox = 0

        """
        datasplit是划分的训练集/验证集文件名 annos是标注
        """
        datasplit = set(utils.file_lines(im_list))
        annos = utils.load_json(anno_file)

        for i, line in enumerate(annos):
            utils.progress_bar(i, len(annos), 'Load Anno...')
            """
                line =
                    {'name': '223_89_t20201125085855802_CAM3.jpg',
                    'image_height': 3500,
                    'image_width': 4096,
                    'category': 4,
                    'bbox': [1702.79, 2826.53, 1730.79, 2844.53]}
            """
            name = line['name']
            image_height = line['image_height']
            image_width = line['image_width']
            label = line['category']
            bbox = line['bbox']  # xyxy

            if name not in datasplit:
                continue

            counter[class_names[label]] += 1
            tot_bbox += 1

            abspath = os.path.abspath(os.path.join(image_root, name))
            # abspath = '/home/raid/xhy/scp079/tile/datasets/tile/tile_round1_train_20201231/train_imgs/245_141_t20201128145004324_CAM1.jpg'
            self.image_paths.add(abspath)
            self.bboxes[name].append(bbox)
            self.labels[name].append(label)

        self.image_paths = list(self.image_paths)
        self.image_paths.sort()

        for name in class_names:
            utils.color_print(f'{name}: {counter[name]} ({counter[name]/tot_bbox*100:.2f}%)', 5)
        
        utils.color_print(f'Total bboxes: {tot_bbox}', 4)        
        
        self.format = format

        assert transforms is not None, '"transforms" is required'

        self.transforms = transforms
        self.max_size = max_size

        ### ===========WARNING: DEL BLOW===============
        # for k, v in self.labels.items():
        #     if len(v) == 0:
        #         print(k)

        # import ipdb
        # ipdb.set_trace()
        ### ===========WARNING: DEL ABOVE===============

    def load_image_and_boxes(self, index):
        image_path = self.image_paths[index]
        name = os.path.basename(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'{image_path} not found.')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # 转成0~1之间

        bboxes = np.array(self.bboxes[name])
        labels = np.array(self.labels[name])

        h, w, _ = image.shape

        return image, bboxes, labels, image_path, (w, h)

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {'image': image,
             'bboxes': bboxes,
             'label': label,
             'path': path,
             'yolo_boxes': yolo_boxes,
             'yolo4_boxes': yolo4_boxes,
             'yolo5_boxes': yolo5_boxes
            }

        """
        image, bboxes, labels, image_path, (org_w, org_h) = self.load_image_and_boxes(index)

        if len(bboxes) == 0:
            raise RuntimeError(f'no bboxes found in {image_path}.')

        for i in range(20):
            sample = self.transforms(**{
                'image': image,
                'bboxes': bboxes,
                'labels': labels
            })

            if len(sample['bboxes']) > 0:
                break

        sample['bboxes'] = torch.Tensor(sample['bboxes']) 
        sample['labels'] = torch.Tensor(sample['labels'])  # <--- add this!
        sample['path'] = image_path

        sample.update(voc_to_yolo_format(sample, opt))

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.image_paths))

        return len(self.image_paths)




class TILETestDataset(dataset.Dataset):
    """TILE Dataset for training and validation.

    Args:
        voc_root(str): root dir to dataset
        class_names(list(str)): class names
        split(str): .txt file in ImageSets/Main
        transforms(albumentations.transform): required, input images and bboxes will be applied simultaneously
        max_size(int): maximum data returned
    """

    def __init__(self, voc_root, transforms=None, max_size=None):
        voc_root = os.path.join(voc_root, 'tile_round1_testA_20201231')

        utils.color_print(f'Use dataset: {voc_root}, split: test', 3)

        image_root = f'{voc_root}/testA_imgs'

        self.image_paths = []

        image_paths = os.listdir(image_root)

        for path in image_paths:
            abspath = os.path.abspath(os.path.join(image_root, path))
            self.image_paths.append(abspath)

        utils.color_print(f'Total images: {len(self.image_paths )}', 4)        
        
        self.format = format

        assert transforms is not None, '"transforms" is required'

        self.transforms = transforms
        self.max_size = max_size

    def load_image(self, index):
        image_path = self.image_paths[index]
        name = os.path.basename(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'{image_path} not found.')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # 转成0~1之间

        h, w, _ = image.shape

        return image, image_path

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {'image': image,
             'bboxes': bboxes,
             'label': label,
             'path': path,
             'yolo_boxes': yolo_boxes,
             'yolo4_boxes': yolo4_boxes,
             'yolo5_boxes': yolo5_boxes
            }

        """
        image, image_path = self.load_image(index)

        sample = self.transforms(**{
            'image': image,
        })

        sample['path'] = image_path
        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.image_paths))

        return len(self.image_paths)