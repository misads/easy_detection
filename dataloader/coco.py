import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from options import opt
from pycocotools.coco import COCO
import cv2

from dataloader.data_helper import voc_to_yolo_format

def coco_90_to_80_classes(id):
    m = [0,1,2,3,4,5,6,7,8,9,10,11,0,12,13,14,15,16,17,18,19,20,21,22,23,24,0,
            25,26,0,0,27,28,29,30,31,32,33,34,35,36,37,38,39,40, 0,41,42,43,44,45,46,
            47,48,49,50,51,52,53,54,55,56,57,58,59,60, 0,61,0,0,62,0,63,64,65,66,67,68,69,
            70,71,72,73,0,74,75,76,77,78,79,80] 

    return m[id]

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name, transforms=None):
        """
        Args:
            root_dir (string): COCO directory.
            transforms (callable, optional): Optional transforms to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transforms = transforms

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img, path = self.load_image(idx)
        annot = self.load_annotations(idx)
        bboxes = annot[:, :4]
        labels = annot[:, 4]

        if self.transforms:           
            for i in range(20):
                sample = self.transforms(**{
                    'image': img,
                    'bboxes': bboxes,
                    'labels': labels
                })

                if len(sample['bboxes']) > 0:
                    break

        sample['bboxes']= torch.Tensor(sample['bboxes'])
        sample['labels']= torch.Tensor(sample['labels'])
        sample['path'] = path

        sample.update(voc_to_yolo_format(sample, opt))  # yolo format

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #
        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0 , path

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80
