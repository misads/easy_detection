import argparse
from misc_utils import load_json
import misc_utils as utils
from mscv.summary import *
import os
import cv2
from vis import visualize_boxes
import numpy as np
from collections import defaultdict

npa = np.array

img_root = '/home/raid/public/datasets/tile/tile_round1_testA_20201231/testA_imgs/'

writer = create_summary_writer('vis_json')

cateNames = ['_bkg', 'edge', 'corner', 'whitespot', 'lightblock', 'darkblock', 'aperture']

def parse_args():
    # 创建一个parser对象
    parser = argparse.ArgumentParser(description='parser demo')
 
    # str类型的参数(必填)（不需要加-标志）
    parser.add_argument('path')
 
    # 选择类型的参数
    # parser.add_argument('--load', nargs='+')
 

 
    args = parser.parse_args()
 
    return args
    
opt = parse_args()

results = load_json(opt.path)

bboxes = defaultdict(list) 
labels = defaultdict(list) 
scores = defaultdict(list) 

for result in results:
    name = result['name']
    label = result['category']
    bbox = result['bbox']
    score = result['score']
    bboxes[name].append(bbox)
    labels[name].append(label)
    scores[name].append(score)

for i, name in enumerate(scores.keys()):
    utils.progress_bar(i, len(scores.keys()))
    path = os.path.join(img_root, name)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualize_boxes(image=image, boxes=npa(bboxes[name]), labels=npa(labels[name]), probs=npa(scores[name]), class_labels=cateNames)
    write_image(writer, 'test', f'result_{i}', image, 0, 'hwc')

writer.flush()