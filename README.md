# easy_detection


　　COCO和VOC目标检测，基于pytorch，开箱即用，**不需要CUDA编译**。支持Faster_RCNN、Cascade_RCNN、Yolo系列、SSD。


  
　　对比mmdetection，mmdetection功能很多，但是封装的层数也过多，对于初学者不是太友好。因此将经典的检测模型用简单的方式整理或重写了一下。如果遇到问题欢迎提issue或者与我联系。


## 介绍

　　支持结果可视化、自定义数据集、多卡同步训练。  
  
　　训练时间(Faster RCNN)：

||单卡2080ti|8卡2080ti|
|---|---|---|
|VOC|6小时|45分钟|
|COCO|48小时|6小时|


## 使用说明

安装和使用说明见 [使用手册.md](https://github.com/misads/easy_detection/blob/master/_assets/_docs/get_started.md).



## 模型指标和预训练模型

### VOC数据集

| 结构 | mAP@.5 | 下载链接 | 密码 | sha256 |
| ----------- | -------- | ----- | ----- | ----- |
| YoloV2  | 76.46|   [[百度网盘]](https://pan.baidu.com/s/1UyWGG1kn5h1l_FHP3idurw)| mwik | 5d29a34b |
| FasterRCNN + Res50 + FPN | 82.39 |  [[百度网盘]](https://pan.baidu.com/s/17NDNGeVRYxCG0vWqgaFDxQ) | isqt | 3d5c3b15 |
| CascadeRCNN + Res50 + FPN | 81.90 |  - | - | - |
| SSD300 + VGG16 | 79.21 | [[百度网盘]](https://pan.baidu.com/s/18XN0Atybz27DnwFdUsMRPg)| 59y0 | 106c0fc9 |
| SSD512 + VGG16 |   82.14 | [[百度网盘]](https://pan.baidu.com/s/1CYB7GvLYxin01Oqwo0v7ZQ)| 0iur | 844b40b3 |



### COCO数据集


| 结构 | COCO AP\* | mAP@.5 | mAP@.75 |下载链接 | 密码 | sha256 |
| --------------- | ---------- | ------ | -------- | ----- | ----- | ----- |
| FasterRCNN + Res50 + FPN | 35.41 |57.11| 38.43 | [[pytorch]](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) | - | 258fb6c6 |
| CascadeRCNN + Res50 + FPN | 38.71 |56.61| 42.16 | - | - | - |
| YoloV3  | - | 55.3| - | [[百度网盘]](https://pan.baidu.com/s/1SxmjpgCbwAEyRtwLNhG3xQ) | cf4j | 943b926a|
| YoloV4 | - | 62.8 |- | [[百度网盘]](https://pan.baidu.com/s/1keDDPyMvpX11jnXbJsoTrg) | nio7 | 797dc954 |
| YoloV5 |  - |  64.30 |- | [[百度网盘]](https://pan.baidu.com/s/1j45qGCEu5_Tl0BlDF8ixnw) | cssw | 8e54a2e8 |

　　\*注：COCO AP是IoU@\[0.5:0.95\]的mAP平均值。

## Reference

- SSD <https://github.com/lufficc/SSD>
  
- YoloV2、YoloV3 <https://github.com/andy-yun/pytorch-0.4-yolov3>

- EfficientDet <https://github.com/rwightman/efficientdet-pytorch>

- YoloV4 <https://github.com/Tianxiaomo/pytorch-YOLOv4> <https://github.com/argusswift/YOLOv4-pytorch>

- YoloV5 <https://github.com/ultralytics/yolov5>

- Faster_RCNN <https://github.com/pytorch/vision/tree/master/torchvision/models/detection>

- RetinaNet <https://github.com/yhenon/pytorch-retinanet>
