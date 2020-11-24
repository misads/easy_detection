## 环境需求

```yaml
python >= 3.6
numpy >= 1.16.0
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
mscv >= 0.0.3
albumentations >= 0.4.0
torch-template >= 0.0.4
opencv-python >= 4.0.0.21
timm == 0.1.30  # timm >= 0.2.0 
typing_extensions == 3.7.2
tqdm >= 4.49.0
pycocotools == 2.0

```
都是很好装的包，不需要编译。

## 训练和验证模型

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


## 添加新的模型：


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