# easy_detection

　　easy_detection是一个能够轻松上手的Pytorch目标检测框架，它不需要cuda编译，支持Faster_RCNN、Yolo系列(v2~v5)、EfficientDet等经典网络，能够一键预览数据集标注并对训练过程进行可视化，同时有配套的web平台进行任务监控和管理。

![preview](http://www.xyu.ink/wp-content/uploads/2020/10/COCO2.png)

## Features

- 可视化
  - [x] 数据集标注可视化
  - [x] 训练任务web端可视化管理
  - [x] 预测结果可视化输出

- 数据格式
  - [x] VOC
  - [x] COCO
  - [x] 自定义格式

- 网络模型
  - [x] Faster-RCNN
  - [x] Cascade-RCNN
  - [x] YoloV2、V3  
  - [x] YoloV4  
  - [x] YoloV5  
  - [x] SSD300、SSD512
  - [x] EfficientDet
  - [x] RetinaNet
  
- 预测框融合
  - [x] 多尺度融合
  - [x] nms
  - [x] Weighted Box Fusion(WBF)

- Scheduler
  - [x] Step Scheduler
  - [x] Cos Scheduler

- Metrics
  - [x] mAP


## 安装和使用教程

安装和使用方法见 [使用手册.md](https://github.com/misads/easy_detection/blob/master/_assets/_docs/get_started.md).



## 预训练模型

### VOC数据集

| 结构 | mAP@.5 | 下载链接 | 密码 | sha256 |
| ----------- | -------- | ----- | ----- | ----- |
| YoloV2  | 76.46|   [[百度网盘]](https://pan.baidu.com/s/1UyWGG1kn5h1l_FHP3idurw)| mwik | 5d29a34b |
| FasterRCNN + Res50 + FPN | 83.26 |  [[百度网盘]](https://pan.baidu.com/s/17NDNGeVRYxCG0vWqgaFDxQ) | isqt | 3d5c3b15 |
| SSD300 + VGG16 | 79.21 | [[百度网盘]](https://pan.baidu.com/s/18XN0Atybz27DnwFdUsMRPg)| 59y0 | 106c0fc9 |
| SSD512 + VGG16 |   82.14 | [[百度网盘]](https://pan.baidu.com/s/1CYB7GvLYxin01Oqwo0v7ZQ)| 0iur | 844b40b3 |



### COCO数据集


| 结构 | mAP@(.5:0.95) | mAP@.5 | 下载链接 | 密码 | sha256 |
| ----------- | ---------- | -------- | ----- | ----- | ----- |
| YoloV3  | - | 55.3| [[百度网盘]](https://pan.baidu.com/s/1SxmjpgCbwAEyRtwLNhG3xQ) | cf4j | 943b926a|
| FasterRCNN+Res50+FPN | 35.5 |57.9|  [[pytorch]](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) | - | 258fb6c6 |
| YoloV4 | - | 62.8 | [[百度网盘]](https://pan.baidu.com/s/1keDDPyMvpX11jnXbJsoTrg) | nio7 | 797dc954 |
| YoloV5 |  - |  64.30 | [[百度网盘]](https://pan.baidu.com/s/1j45qGCEu5_Tl0BlDF8ixnw) | cssw | 8e54a2e8 |


## Reference

- SSD <https://github.com/lufficc/SSD>
  
- YoloV2、YoloV3 <https://github.com/andy-yun/pytorch-0.4-yolov3>

- EfficientDet <https://github.com/rwightman/efficientdet-pytorch>

- YoloV4 <https://github.com/Tianxiaomo/pytorch-YOLOv4> <https://github.com/argusswift/YOLOv4-pytorch>

- YoloV5 <https://github.com/ultralytics/yolov5>

- Faster_RCNN <https://github.com/pytorch/vision/tree/master/torchvision/models/detection>

- RetinaNet <https://github.com/yhenon/pytorch-retinanet>
