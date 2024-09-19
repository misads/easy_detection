# easy_detection

　　COCO和VOC目标检测，基于pytorch，开箱即用，不需要CUDA编译。支持Faster_RCNN、Cascade_RCNN、Yolo系列、SSD。

　　对比mmdetection，mmdetection功能很多，但是封装的层数也过多，对于初学者不是太友好。因此将经典的检测模型用简单的方式整理或重写了一下。如果遇到问题欢迎提issue或者与我联系。

　　Faster RCNN实现细节可以参考我的Blog：[Faster RCNN实现细节](http://wiki.xyu.ink/#/cv/faster_rcnn?id=a31-faster-rcnn).

## 目录
- [介绍](#介绍)
- [使用说明](#使用说明)
- [模型指标和预训练模型](#模型指标和预训练模型)
- [参考链接](#参考链接)

## 介绍

　　支持结果可视化、自定义数据集、多卡同步训练。  

　　训练时间(Faster RCNN)：

||单卡2080ti|8卡2080ti|
|---|---|---|
|VOC|6小时|45分钟|
|COCO|48小时|6小时|

## 使用说明

- 安装和使用方法见 [使用手册.md](/_assets/_docs/tutorial.md).

- Faster RCNN实现细节请参考 [Faster RCNN实现细节](http://wiki.xyu.ink/#/cv/faster_rcnn?id=a31-faster-rcnn).

- 带有详细注释的代码细节请参考 `network/Faster_RCNN_v2/faster_rcnn`目录下的相关文件。

## 模型指标和预训练模型

### VOC数据集

| 结构 | mAP@.5 | 下载链接 | 密码 | 日志 |
| ----------- | -------- | ----- | ----- | ----- |
| YoloV2  | 76.46|   [[百度网盘]](https://pan.baidu.com/s/1UyWGG1kn5h1l_FHP3idurw)| mwik | [[训练日志]](/_assets/_logs/yolo2_voc.txt) |
| FasterRCNN + Res50 + FPN | 82.39 |  [[百度网盘]](https://pan.baidu.com/s/17NDNGeVRYxCG0vWqgaFDxQ) | isqt |  [[训练日志]](/_assets/_logs/faster_rcnn_voc.txt) |
| CascadeRCNN + Res50 + FPN | 81.90 |  - | - | [[训练日志]](/_assets/_logs/cascade_rcnn_voc.txt)  |
| SSD300 + VGG16 | 79.21 | [[百度网盘]](https://pan.baidu.com/s/18XN0Atybz27DnwFdUsMRPg)| 59y0 | - |
| SSD512 + VGG16 |   82.14 | [[百度网盘]](https://pan.baidu.com/s/1CYB7GvLYxin01Oqwo0v7ZQ)| 0iur | - |



### COCO数据集


| 结构 | COCO AP\* | mAP@.5 | mAP@.75 |下载链接 | 日志 |
| --------------- | ---------- | ------ | ----- | ----- | ----- |
| FasterRCNN + Res50 + FPN | 35.41 |57.11| 38.43 | [[pytorch]](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) | [[训练日志]](/_assets/_logs/faster_rcnn_coco.txt) |
| CascadeRCNN + Res50 + FPN | 38.71 |56.61| 42.16 | - | [[训练日志]](/_assets/_logs/cascade_rcnn_coco.txt) |


　　\*注：COCO AP是IoU@\[0.5:0.95\]的mAP平均值。

## 参考链接

- SSD <https://github.com/lufficc/SSD>
  
- YoloV2、YoloV3 <https://github.com/andy-yun/pytorch-0.4-yolov3>

- EfficientDet <https://github.com/rwightman/efficientdet-pytorch>

- YoloV4 <https://github.com/Tianxiaomo/pytorch-YOLOv4> <https://github.com/argusswift/YOLOv4-pytorch>

- YoloV5 <https://github.com/ultralytics/yolov5>

- Faster_RCNN <https://github.com/pytorch/vision/tree/master/torchvision/models/detection>

- RetinaNet <https://github.com/yhenon/pytorch-retinanet>
