# detection_template

　　一个目标检测的通用框架(不需要cuda编译)，支持Yolo全系列(v2~v5)、EfficientDet、RetinaNet、Cascade-RCNN等SOTA网络，Kaggle小麦检测12/2245。

![preview](http://www.xyu.ink/wp-content/uploads/2020/10/COCO2.png)

## Functionalities

- 数据格式
  - [x] VOC
  - [ ] CSV文件
  - [x] COCO

- 网络模型
  - [x] EfficientDet(目前不支持训练过程中验证)
  - [x] YoloV2、V3  
  - [x] YoloV4  
  - [x] YoloV5  
  - [x] SSD300、SSD512(目前只支持vgg backbone,且不支持预训练模型)
  - [x] Faster-RCNN
  - [ ] Cascade-RCNN
  - [x] RetinaNet
  
- TTA
  - [ ] 多尺度融合
  - [x] nms
  - [x] Weighted Box Fusion(WBF)
  - [ ] 伪标签

- Scheduler
  - [ ] 验证集指标不下降时学习率衰减

- Metrics
  - [x] mAP

- 可视化
  - [x] 数据集bbox预览
  - [x] dataloader数据增强预览
  - [x] 预测结果预览

- 辅助工具
  - [ ] 手工标注工具


## 安装和使用教程

安装和使用教程见 [get_started.md](https://github.com/misads/detection_template/blob/master/_assets/_docs/get_started.md).



## 预训练模型

| Model | backbone | 数据集 | 论文mAP@.5 | 复现mAP@.5 | 下载链接 | 密码 | sha256 | 
| ----- | ------ | -------- | ---------- | -------- | ----- | ----- | ----- |
| YoloV2 | Darknet-19 | VOC |76.8|76.46|   [[百度网盘]](https://pan.baidu.com/s/1UyWGG1kn5h1l_FHP3idurw)| mwik | 5d29a34b |
| YoloV3 | Darknet-19 | COCO |55.3|-| [[百度网盘]](https://pan.baidu.com/s/1SxmjpgCbwAEyRtwLNhG3xQ) | cf4j | 943b926a|
| FRCNN | Res50+FPN | VOC | - |83.26 |  [[百度网盘]](https://pan.baidu.com/s/17NDNGeVRYxCG0vWqgaFDxQ) | isqt | 3d5c3b15 |
| FRCNN | Res50+FPN |  COCO |  |48.81|  |  |  |
| YoloV4 | CSPDarknet-53 | COCO| 62.8 | - | [[百度网盘]](https://pan.baidu.com/s/1keDDPyMvpX11jnXbJsoTrg) | nio7 | 797dc954 |
| YoloV5 | CSPDarknet | COCO | - |  64.30 | [[百度网盘]](https://pan.baidu.com/s/1j45qGCEu5_Tl0BlDF8ixnw) | cssw | 8e54a2e8 |

## Reference

- SSD <https://github.com/lufficc/SSD>
  
- YoloV2、YoloV3 <https://github.com/andy-yun/pytorch-0.4-yolov3>

- EfficientDet <https://github.com/rwightman/efficientdet-pytorch>

- YoloV4 <https://github.com/Tianxiaomo/pytorch-YOLOv4> <https://github.com/argusswift/YOLOv4-pytorch>

- YoloV5 <https://github.com/ultralytics/yolov5>

- Faster_RCNN <https://github.com/pytorch/vision/tree/master/torchvision/models/detection>

- RetinaNet <https://github.com/yhenon/pytorch-retinanet>
