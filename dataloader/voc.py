# encoding=utf-8
import ipdb

import torch
import torchvision.transforms.functional as F
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms
import albumentations as A

import xml.etree.ElementTree as ET
import misc_utils as utils
import random
import numpy as np
import cv2


class VOCTrainValDataset(dataset.Dataset):
    """VOC Dataset for training.

    Args:
        voc_root(str): root dir to voc dataset
        class_names(list(str)): class names
        split(str): .txt file in ImageSets/Main
        format(str): 'jpg' or 'png'
        transforms(albumentations.transform): required, input images and bboxes will be applied simultaneously
        max_size(int): maximum data returned

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

    def __init__(self, voc_root, class_names, split='train.txt', format='jpg', transforms=None, max_size=None):
        im_list = os.path.join(voc_root, f'ImageSets/Main/{split}')
        image_root = os.path.join(voc_root, 'JPEGImages')

        self.image_paths = []
        self.bboxes = []
        self.labels = [] 

        with open(im_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_id = line.rstrip('\n')
                abspath = os.path.abspath(os.path.join(image_root, f'{image_id}.{format}'))
                self.image_paths.append(abspath)
                with open(os.path.join(voc_root, f'Annotations/{image_id}.xml'), 'r') as anno:
                    tree = ET.parse(anno)

                # 解析xml标注
                root = tree.getroot()
                bboxes = []
                labels = []
                for obj in root.iter('object'):  # 多个元素
                    # difficult = obj.find('difficult').text
                    class_name = obj.find('name').text
                    if class_name not in class_names:
                        raise Exception(f'"{class_name}" not in class names({class_names}).')
                    class_id = class_names.index(class_name)
                    bbox = obj.find('bndbox')
                    x1 = int(bbox.find('xmin').text)
                    y1 = int(bbox.find('ymin').text)
                    x2 = int(bbox.find('xmax').text)
                    y2 = int(bbox.find('ymax').text)
                    bboxes.append([x1,y1,x2,y2])
                    labels.append(class_id)

                self.bboxes.append(bboxes)
                self.labels.append(labels)

        self.format = format

        assert transforms is not None, '"transforms" is required'

        self.transforms = transforms
        self.max_size = max_size


    def load_image_and_boxes(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # 转成0~1之间

        bboxes = np.array(self.bboxes[index])
        labels = np.array(self.labels[index])

        return image, bboxes, labels, image_path

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {'image': image,
             'bboxes': bboxes,
             'label': label,
             'path': path
            }

        """
        image, bboxes, labels, image_path = self.load_image_and_boxes(index)
        target = {}
        # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        yolo_boxes = np.zeros([50, 5])

        for i in range(10):
            sample = self.transforms(**{
                'image': image,
                'bboxes': bboxes,
                'labels': labels
            })
            
            sample['bboxes'] = torch.Tensor(sample['bboxes'])

            if len(sample['bboxes']) > 0:
                bboxes = sample['bboxes']
                for i, bbox in enumerate(bboxes):
                    if i >= 50:
                        break

                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    c_x, c_y = x1 + w / 2, y1 + h / 2
                    w, c_x = w / 512, c_x / 512
                    h, c_y = h / 512, c_y / 512

                    yolo_boxes[i, :] = labels[i], c_x, c_y, w, h  # 中心点坐标、宽、高

                """
                注意!! yxyx 
                """
                sample['bboxes'][:,[0,1,2,3]] = sample['bboxes'][:,[1,0,3,2]]  
                break
                
        sample['labels'] = torch.Tensor(sample['labels'])  # <--- add this!
        sample['path'] = image_path

        sample['yolo_boxes'] = torch.Tensor(yolo_boxes).view([-1])
        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.image_paths))

        return len(self.image_paths)