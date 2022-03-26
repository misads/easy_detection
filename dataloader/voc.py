# encoding=utf-8
import ipdb

import torch
import torchvision.transforms.functional as F
import os
from PIL import Image
from torchvision import transforms
import torch.utils.data.dataset as dataset
import albumentations as A

import misc_utils as utils
import xml.etree.ElementTree as ET
import misc_utils as utils
import random
import pickle
import numpy as np
import cv2

from collections import defaultdict
from dataloader.data_helper import voc_to_yolo_format

limit = lambda x, minimum, maximum: min(max(minimum, x), maximum)


class VOCTrainValDataset(dataset.Dataset):
    """VOC Dataset for training.

    Args:
        voc_root(str): root dir to voc dataset
        class_names(list(str)): class names
        split(str): .txt file in ImageSets/Main
        format(str): 'jpg' or 'png'
        transforms(albumentations.transform): required, input images and bboxes will be applied simultaneously
        max_size(int): maximum data returned
        use_cache(bool): whether use cached pickle file or load all xml annotations

    Example:
        import albumentations as A
        val_transform =A.Compose(  # images and bboxes will transform together
            [
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ], 
            p=1.0, 
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0, 
                min_visibility=0,
                label_fields=['labels']
            )
        )

        train_dataset = VOCTrainValDataset('voc', ['person'], 'train.txt', transforms=val_transform)

        for i, sample in enumerate(train_dataset):
            image, bboxes, labels, path = sample['image'], sample['bboxes'], sample['labels'], sample['path']
            

    """

    def __init__(self, voc_root, class_names, split='train.txt', format='jpg', transforms=None, max_size=None, use_cache=False, use_difficult=False, first_gpu=True):
        if first_gpu:
            utils.color_print(f'Use dataset: {voc_root}, split: {split[:-4]}', 3)

        im_list = os.path.join(voc_root, f'ImageSets/Main/{split}')
        image_root = os.path.join(voc_root, 'JPEGImages')

        self.image_paths = []
        self.bboxes = []
        self.labels = [] 
        self.difficults = []

        counter = defaultdict(int)
        tot_bbox = 0
        difficult_bbox = 0

        """
        如果有缓存的pickle文件，就直接从pickle文件读取bboxes
        """
        os.makedirs('.cache', exist_ok=True)

        cache_file = os.path.join('.cache', f'{os.path.basename(voc_root)}_{split[:-4]}.pkl')
        if use_cache and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            
            if first_gpu:
                utils.color_print(f'Use cached annoations.', 3)

            self.image_paths, self.bboxes, self.labels, self.difficults, \
            counter, tot_bbox, difficult_bbox = data
            
        else:  # 没有缓存文件
            with open(im_list, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if first_gpu:
                        utils.progress_bar(i, len(lines), 'Load Anno...')
                    
                    image_id = line.rstrip('\n')
                    if not os.path.isfile(os.path.join(voc_root, f'Annotations/{image_id}.xml')):
                        continue
                    abspath = os.path.abspath(os.path.join(image_root, f'{image_id}.{format}'))
                    self.image_paths.append(abspath)
                    with open(os.path.join(voc_root, f'Annotations/{image_id}.xml'), 'r') as anno:
                        tree = ET.parse(anno)

                    # 解析xml标注
                    root = tree.getroot()
                    bboxes = []
                    labels = []

                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)

                    for obj in root.iter('object'):  # 多个元素
                        # difficult = obj.find('difficult').text
                        class_name = obj.find('name').text

                        difficult = obj.find('difficult').text
                        if difficult != '0' and not use_difficult: 
                            difficult_bbox += 1
                            continue  # 忽略困难样本

                        if class_name not in class_names:
                            continue  # class_names中没有的类别是忽略还是报错
                            raise Exception(f'"{class_name}" not in class names({class_names}).')
                            
                        class_id = class_names.index(class_name)
                        bbox = obj.find('bndbox')
                        x1 = limit(int(bbox.find('xmin').text), 0, width)
                        y1 = limit(int(bbox.find('ymin').text), 0, height)
                        x2 = limit(int(bbox.find('xmax').text), 0, width)
                        y2 = limit(int(bbox.find('ymax').text), 0, height)

                        if x2 - x1 <= 2 or y2 - y1 <= 2:  # 面积很小的标注
                            continue

                        counter[class_name] += 1
                        tot_bbox += 1
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

                    self.bboxes.append(bboxes)
                    self.labels.append(labels)

            """
            存放到缓存文件
            """
            data = [self.image_paths, self.bboxes, self.labels, self.difficults, \
                counter, tot_bbox, difficult_bbox]

            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

        if first_gpu:
            for name in class_names:
                utils.color_print(f'{name}: {counter[name]} ({counter[name]/tot_bbox*100:.2f}%)', 5)
        
            utils.color_print(f'Total bboxes: {tot_bbox}', 4)
            if difficult_bbox:
                utils.color_print(f'{difficult_bbox} difficult bboxes ignored.', 1)
        
        self.format = format

        assert transforms is not None, '"transforms" is required'

        self.transforms = transforms
        self.max_size = max_size


    def load_image_and_boxes(self, index):
        image_path = self.image_paths[index]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'{image_path} not found.')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # 转成0~1之间

        bboxes = np.array(self.bboxes[index])
        labels = np.array(self.labels[index])

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
            pass

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

        sample.update(voc_to_yolo_format(sample))

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.image_paths))

        return len(self.image_paths)
