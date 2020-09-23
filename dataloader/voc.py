# encoding=utf-8
import ipdb

import torch
import torchvision.transforms.functional as F
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms

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
        split(str): txt in ImageSets/Main
        format(str): 'jpg' or 'png'
        scale(int): images will be first resize to this size
        aug(bool): data argument (×8)
        norm(bool): normalization

    Example:
        train_dataset = VOCTrainValDataset('voc')
        for i, data in enumerate(train_dataset):
            input, bboxes, labels = data['input'], data['bboxes'], data['labels']

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

        self.transforms = transforms
        self.max_size = max_size


    def load_image_and_boxes(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # 转成0~1之间

        bboxes = np.array(self.bboxes[index])
        labels = np.array(self.labels[index])

        # records = self.marking[self.marking['image_id'] == image_id]
        # boxes = records[['x', 'y', 'w', 'h']].values
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        # return image, boxes
        return image, bboxes, labels, image_path

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {'input': input,
             'bboxes': bboxes,
             'label': label,
             'path': path
            }

        """
        image, bboxes, labels, image_path = self.load_image_and_boxes(index)
        target = {}
        # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        target['input'] = F.to_tensor(image)
        target['bboxes'] = torch.Tensor(bboxes)

        if self.transforms is not None:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': bboxes,
                    'labels': labels
                })
                
                sample['bboxes'] = torch.Tensor(sample['bboxes'])

                if len(sample['bboxes']) > 0:
                    sample['bboxes'][:,[0,1,2,3]] = sample['bboxes'][:,[1,0,3,2]]  
                    break
                    
            sample['input'] = sample['image']
            sample['labels'] = torch.Tensor(sample['labels'])  # <--- add this!
            sample['path'] = image_path
            return sample
        """
        注意!! yxyx 
        """
        target['bboxes'][:,[0,1,2,3]] = target['bboxes'][:,[1,0,3,2]]  
        target['labels'] = torch.Tensor(labels)  # <--- add this!
        target['path'] = image_path

        yolo_boxes = np.zeros([50, 5])
        bboxes = target['bboxes']
        for i, bbox in enumerate(bboxes):
            if i >= 50:
                break

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            c_x = x1 + w / 2
            c_y = y1 + h / 2
            w, c_x = w / 512, c_x / 512
            h, c_y = h / 512, c_y / 512

            yolo_boxes[i, 0] = labels[i]
            yolo_boxes[i, 1] = c_x
            yolo_boxes[i, 2] = c_y
            yolo_boxes[i, 3] = w
            yolo_boxes[i, 4] = h

        return target

        # if self.scale:
        #     input = F.resize(input, self.scale)
        #     label = F.resize(label, self.scale)

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.image_paths))

        return len(self.image_paths)