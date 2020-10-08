# detection_template

一个目标检测的Baseline(不需要cuda编译)，Kaggle小麦检测12/2245。

![preview](http://www.xyu.ink/wp-content/uploads/2020/06/cityscapes.png)

## Todo List

- 数据格式
  - [x] VOC
  - [ ] CSV文件
  - [ ] COCO(RetinaNet代码里有)

- 网络模型
  - [x] EfficientDet(目前不支持训练过程中验证)
  - [x] YoloV2、V3
  - [ ] YoloV4
  - [x] YoloV5
  - [x] SSD300、SSD512(目前只支持vgg backbone,且不支持预训练模型)
  - [x] Faster-RCNN
  - [ ] Cascade-RCNN
  - [x] RetinaNet
  
- TTA
  - [ ] 多尺度融合
  - [x] nms
  - [ ] Weighted Box Fusion
  - [ ] 伪标签

- Scheduler
  - [ ] 验证集指标不下降时学习率衰减

- Metrics
  - [x] mAP

- 可视化
  - [x] 数据集bbox预览
  - [x] dataloader数据增强预览
  - [ ] 预测结果预览

- 辅助工具
  - [ ] 手工标注工具

## Prerequisites

```yaml
python >= 3.6
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
mscv >= 0.0.3
albumentations >= 0.4.0
```

## Code Usage

```bash
Code Usage:
Training:
    python train.py --tag your_tag --model Yolo2 --epochs 200 -b 3 --lr 0.0001 --gpu 0

Resume Training (or fine-tune):
    python train.py --tag your_tag --model Yolo2 --epochs 20 -b 2 --load checkpoints/your_tag/9_Effdet.pt --resume --gpu 0

Eval:
    python eval.py --model Yolo2 -b 2 --load checkpoints/your_tag/9_Yolo2.pt --gpu 1 --vis

Generate Submission:
    python submit.py --model Yolo2 --load checkpoints/your_tag/9_Yolo2.pt -b 2 --gpu 0

See Running Log:
    cat logs/your_tag/log.txt

Clear(delete all files with the tag, BE CAREFUL to use):
    python clear.py your_tag

See ALL Running Commands:
    cat run_log.txt
```

## 如何添加新的模型：


1、复制`network`目录下的`Default`文件夹，改成另外一个名字(比如`MyNet`)。

2、在`network/__init__.py`中`import`你模型的`Model`类并且在`models = {}`中添加它。
```python
from MyNet.Model import Model as MyNet
models = {
    'default': Default,
    'MyNet': MyNet,
}
```

3、尝试 `python train.py --model MyNet` 看能否成功运行


## 如何训练自己的数据集

1、将自己的数据集制作成VOC格式；

2、在datasets目录一下建立数据集根目录的软链接：
```bash
cd datasets
ln -s /home/<abspath> mydataset
```

3、修改dataloader目录下的`dataloaders.py`，需要修改的有数据集目录`voc_root`，类别的列表`class_names`以及训练和验证时的`transforms`。

