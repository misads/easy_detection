# detection_template

一个目标检测的Baseline(不需要cuda编译)，Kaggle小麦检测12/2245。

![preview](http://www.xyu.ink/wp-content/uploads/2020/06/cityscapes.png)

## Todo List

- 数据格式
  - [x] VOC
  - [ ] CSV文件
  - [ ] COCO

- 网络模型
  - [x] EfficientDet(目前不支持训练过程中验证)
  - [x] YoloV2、V3
  - [ ] YoloV4
  - [x] YoloV5
  - [x] SSD300、SSD512(目前只支持vgg backbone,且不支持预训练模型)
  - [x] Faster-RCNN
  - [ ] Cascade-RCNN
  - [ ] RetinaNet
  
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
    python train.py --tag your_tag --model Effdet --epochs 200 -b 3 --lr 0.0001 --gpu 0

Resume Training (or fine-tune):
    python train.py --tag your_tag --model Effdet --epochs 20 -b 2 --load checkpoints/your_tag/9_Effdet.pt --resume --gpu 0

Eval:
    python eval.py --model Effdet -b 2 --load checkpoints/your_tag/9_Effdet.pt --gpu 1

Generate Submission:
    python submit.py --model Effdet --load checkpoints/your_tag/9_Effdet.pt -b 2 --gpu 0

See Running Log:
    cat logs/your_tag/log.txt

Clear(delete all files with the tag, BE CAREFUL to use):
    python clear.py --tag your_tag

See ALL Running Commands:
    cat run_log.txt
```

## 如何添加新的模型：

```
如何添加新的模型：

① 复制network目录下的Default文件夹，改成另外一个名字(比如MyNet)。

② 在network/__init__.py中import你的Model并且在models = {}中添加它。
    from MyNet.Model import Model as MyNet
    models = {
        'default': Default,
        'MyNet': MyNet,
    }

③ 尝试 python train.py --model MyNet 看能否成功运行
```
