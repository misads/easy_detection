# detection_template


一个目标检测的Baseline。

## Todo List

- 数据格式
  - [x] VOC
  - [ ] CSV文件
  - [ ] COCO

- 网络模型
  - [x] EfficientDet
  - [ ] YoloV2、V3
  - [ ] YoloV5
  - [ ] SSD
  - [ ] Faster-RCNN
  - [ ] Cascade-RCNNN
  - [ ] RetinaNet
  
- TTA
  - [ ] 多尺度融合
  - [ ] nms
  - [ ] Weighted Box Fusion

## Prerequisites

```yaml
python >= 3.6
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
mscv >= 0.0.3
```

## Code Usage

```bash
Code Usage:
Training:
    python train.py --tag your_tag --model FFA --epochs 20 -b 2 --lr 0.0001 --gpu 0

Resume Training (or fine-tune):
    python train.py --tag your_tag --model FFA --epochs 20 -b 2 --load checkpoints/your_tag/9_FFA.pt --resume --gpu 0

Eval:
    python eval.py --model FFA -b 2 --load checkpoints/your_tag/9_FFA.pt --gpu 1

Generate Submission:
    python submit.py --model FFA --load checkpoints/your_tag/9_FFA.pt -b 2 --gpu 0

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