import sys
sys.path.insert(0, '.')

from network.Faster_RCNN_v2.faster_rcnn.anchors_v2 import AnchorGenerator
import torch

scales=(32, 64, 128, 256, 512)
octaves=(1, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0))

anchor_generator = AnchorGenerator(
    scales=scales, 
    aspect_ratios=(0.5, 1.0, 2.0),
    octaves=octaves
)
h = 800
w = 1024
images = torch.randn([2, 3, h, w])

feature_maps = [
    torch.randn([2, 256, h // stride, w // stride]) for stride in [4, 8, 16, 32, 64]
]

sum_feats = sum([feat[0][0].numel() for feat in feature_maps])

anchors = anchor_generator(images, feature_maps)

print(len(anchors[0]))
import ipdb
ipdb.set_trace()