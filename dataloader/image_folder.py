# encoding=utf-8
import pdb

import torchvision.transforms.functional as F
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms

import misc_utils as utils
import random
import numpy as np
import cv2


def paired_cut(img_1: Image.Image, img_2: Image.Image, crop_size):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # r = random.randint(-1, 6)
    # if r >= 0:
    #     img_1 = img_1.transpose(r)
    #     img_2 = img_2.transpose(r)

    i, j, h, w = get_params(img_1, crop_size)
    img_1 = F.crop(img_1, i, j, h, w)
    img_2 = F.crop(img_2, i, j, h, w)

    return img_1, img_2


class FolderTrainValDataset(dataset.Dataset):
    """ImageDataset for training.

    Args:
        datadir(str): dataset root path
        aug(bool): data argument (×8)
        norm(bool): normalization

    Example:
        train_dataset = ImageDataset('train.txt', aug=False)
        for i, data in enumerate(train_dataset):
            input, label = data['input']. data['label']

    """

    def __init__(self, datadir, scale=None, crop=None, aug=True, norm=False, max_size=None):
        self.input_path = os.path.join(datadir, 'input')
        self.label_path = os.path.join(datadir, 'label')

        input_list = list(filter(lambda x: not x.startswith('.'), os.listdir(self.input_path)))
        label_list = list(filter(lambda x: not x.startswith('.'), os.listdir(self.label_path)))

        self.im_names = sorted(input_list)
        self.label_names = sorted(label_list)

        self.trans_dict = {0: Image.FLIP_LEFT_RIGHT, 1: Image.FLIP_TOP_BOTTOM, 2: Image.ROTATE_90, 3: Image.ROTATE_180,
                           4: Image.ROTATE_270, 5: Image.TRANSPOSE, 6: Image.TRANSVERSE}

        if isinstance(scale, int):
            scale = (scale, scale)

        self.scale = scale
        if isinstance(crop, int):
            crop = (crop, crop)

        self.crop = crop
        self.aug = aug
        self.norm = norm
        self.max_size = max_size

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {'input': input,
             'label': label,
             'path': path
            }

        """

        assert self.im_names[index] == self.label_names[index], \
            f'input {self.im_names[index]} and label {self.label_names[index]} not matching.'

        input = Image.open(os.path.join(self.input_path, self.im_names[index])).convert("RGB")
        label = Image.open(os.path.join(self.label_path, self.label_names[index])).convert("RGB")

        if self.crop:
            input, label = paired_cut(input, label, self.crop)

        if self.scale:
            input = F.resize(input, self.scale)
            label = F.resize(label, self.scale)

        r = random.randint(0, 7)
        if self.aug and r != 7:
            input = input.transpose(self.trans_dict[r])
            label = label.transpose(self.trans_dict[r])

        if self.norm:  # 分割可以norm 复原不能norm
            input = F.normalize(F.to_tensor(input), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            input = F.to_tensor(input)
            label = F.to_tensor(label)
        return {'input': input, 'label': label, 'path': self.im_names[index]}

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)


class FolderTestDataset(dataset.Dataset):
    """ImageDataset for test.

    Args:
        file_list(str): dataset path'
        norm(bool): normalization

    Example:
        test_dataset = ImageDataset('test', crop=256)
        for i, data in enumerate(test_dataset):
            input, file_name = data

    """
    def __init__(self, datadir, scale=None, norm=False, max_size=None):
        self.input_path = datadir
        self.norm = norm

        im_list = list(filter(lambda x: not x.startswith('.'), os.listdir(self.input_path)))
        self.im_names = sorted(im_list)

        if isinstance(scale, int):
            scale = (scale, scale)

        self.scale = scale
        self.norm = norm
        self.max_size = max_size

    def __getitem__(self, index):

        input = Image.open(os.path.join(self.input_path, self.im_names[index])).convert("RGB")

        if self.scale:
            input = F.resize(input, self.scale)

        if self.norm:
            input = F.normalize(F.to_tensor(input), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            input = F.to_tensor(input)

        return {'input': input, 'path': self.im_names[index]}

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)


def preview_dataset(dataset, path='path'):
    for i, data in enumerate(dataset):
        if i == min(10, len(dataset)):
            break

        c = 'input'
        img = data[c]
        img = np.array(img)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if path in data:
            cv2.putText(img, os.path.basename(data[path]), (1, 35), 0, 1, (255, 255, 255), 2)

        if 'label' in data:
            label = data['label']
            label = np.array(label)
            label = np.transpose(label, (1, 2, 0))
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            cv2.putText(label, 'gt', (1, 35), 0, 1, (255, 255, 255), 2)

            preview = np.hstack((img, label))

        else:
            preview = img

        cv2.imshow('preview', preview)

        cv2.waitKey(0)


if __name__ == '__main__':

    dataset = FolderTrainValDataset('../datasets/Toled', crop=256)
    preview_dataset(dataset)

    dataset = FolderTestDataset('../datasets/Toled/test', scale=256)
    preview_dataset(dataset)




