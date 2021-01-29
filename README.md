# 天池2021广东工业智造创新大赛Faster_RCNN baseline (线上70+)

　　代码非常简单的小白入门级baseline(基于pytorch, 初赛A榜单模70+)。

　　代码开源在: <https://github.com/misads/detection_template/tree/tile>。基于一个非常简单的检测框架[detection_template](https://github.com/misads/detection_template)，模型为Faster_RCNN+ResNet50FPN。线上线下能够很好的对应，欢迎大家来star~。

| 方法                                | 线上分数       | 线下分数       |
| ----------------------------------- | -------------- | -------------- |
| 2000×2000测试，nms阈值为0.5         | ~66            |   66.142564    |
| +nms阈值设为0.1                     | 66.480508      | 66.577477      |
| +4500×3500测试，(需要11G左右显存)   | 67.298847      | 67.581412     |
| +过滤最大得分小于0.6的图像(提升acc) | 69.019690      |  未验证     |
| +多尺度测试（需要19G左右显存）      | 70.019778      |  未验证     |
| + 多模型融合                        | 可以自己试一试 |       |

　　训练方法非常简单，使用Faster_RCNN+ResNet50FPN，将图像crop出`2000×2000`的小块训练（需要11G显存），如果显存足够可以裁出`3000×3000`，没有目标框的小块直接丢弃。（我们也试过resize，由于瑕疵太小效果不是很好）。

　　训练时的`transform`是这样的(代码见`dataloader/transforms/frcnn.py`)：

```python
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

height = width = 2000
train_transform = A.Compose(  # FRCNN
    [
        A.RandomCrop(height=height, width=width, p=1.0),  # 2000×2000
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0),
    ]
)
```

　　测试时将整张图片crop成`2×2`个`4500×3500`的小块，分别测试以后，将结果加上偏移拼接到一起，然后进行阈值为`0.1`的`nms`(代码见`tta.py`)。  

　　为了提高一些`acc`的分数，我们所有检测结果最高置信度小于`0.6`的图片认为没有任何瑕疵，也就是不输出任何结果(代码在`wbf.py`中)。  

　　最后进行多尺度的测试，将`4500×3500`的小块分别放大`1.0`，`1.143`，`1.286`，`1.429`倍后测试，将结果进行`wbf`融合(代码见`wbf.py`)。

## 准备数据

　　下载好数据后，进入`tile_round1_train_20201231`目录，使用下面的代码划分训练集和验证集(会生成一个`train.txt`和一个`val.txt`文件)。

```python
import os
import misc_utils as utils  # 使用 pip install utils-misc 安装
files = os.listdir('train_imgs')
files.sort()

f1 = open('train.txt', 'w')
f2 = open('val.txt', 'w')

a = set()
val = set()

for f in files:
    tileid = '_'.join(f.split('_')[:2])
    a.add(tileid)

val_ratio = 0.2  # 划分20%验证集

for tileid in a:
    if utils.gambling(val_ratio):
        val.add(tileid)
        
for f in files:
    tileid = '_'.join(f.split('_')[:2])
    if tileid in val:
        f2.writelines(f+'\n')
    else:
        f1.writelines(f+'\n')
        
f1.close()
f2.close()

```

## 训练模型

1. 克隆此项目：

```bash
# !-bash
git clone -b tile https://github.com/misads/detection_template
cd detection_template
```

2. 安装下面的依赖项：

```yml
torch>=1.0  # effdet需要torch>=1.5，如果不使用effdet，在network/__init__.py下将其注释掉
tensorboardX>=1.6
utils-misc>=0.0.5
mscv>=0.0.3
matplotlib>=3.1.1
opencv-python>=4.2.0.34  # opencv>=4.4版本需要编译，耗时较长，建议安装4.2版本
opencv-python-headless>=4.2.0.34
albumentations>=0.5.1  # 需要opencv>=4.2
easydict>=1.9
```

　　你也可以安装好`pytorch`后运行`bash install.sh`一键安装。

2. 新建一个`datasets`文件夹：

```bash
mkdir datasets
```

4. 在`datasets`目录中新建一个软链接链接到数据位置：

```python
ln -s <数据下载目录>/tile_round1_train_20201231 datasets/tile
```

5. 运行训练命令：

```bash
python3 train.py --tag frcnn_res50_2k --model Faster_RCNN --scale 2000 --val_freq 20
```

　　参数中的`val_freq`是每`20`代验证一次，线下验证的是**全图**mAP@0.1，mAP@0.3和mAP@0.5的平均值，acc暂不支持验证。  

　　详细的参数使用方法可以参考[说明文档](https://github.com/misads/detection_template/blob/tile/_assets/_docs/get_started.md#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)。

## 验证模型和提交结果

线下验证模型

```bash
python3 eval.py --model Faster_RCNN -b1 --scale 2000 --load checkpoints/frcnn_res50_2k/40_Faster_RCNN.pt --vis
```

　　`--vis`可以加也可以不加，加上后可以在`tensorboard`预览预测的结果，但是**时间会变得很慢**。  

生成提交结果的json：

```bash
python submit.py --model Faster_RCNN -b1 --load checkpoints/frcnn_res50_2k/40_Faster_RCNN.pt --scale 2000
```

去除概率较低的结果，提高acc值：

```bash
wbf.py result.json
```

我们也提供了一个我们训练的模型方便你测试环境和代码的正确性：

　　**40_Faster_RCNN.pt** [百度网盘链接]()

## [进阶]多尺度测试和多模型融合

多尺度测试会将crop的小块放大成不同的尺寸，分别运行以下命令：

```bash
python submit.py --model Faster_RCNN -b1 --load checkpoints/frcnn_res50_2k/40_Faster_RCNN.pt --scale 2000 --ms 1.
```
```bash
python submit.py --model Faster_RCNN -b1 --load checkpoints/frcnn_res50_2k/40_Faster_RCNN.pt --scale 2000 --ms 1.143
```
```bash
python submit.py --model Faster_RCNN -b1 --load checkpoints/frcnn_res50_2k/40_Faster_RCNN.pt --scale 2000 --ms 1.286
```
```bash
python submit.py --model Faster_RCNN -b1 --load checkpoints/frcnn_res50_2k/40_Faster_RCNN.pt --scale 2000 --ms 1.429
```

会生成4个文件：`result.json`，`result_1.143.json`，`result_1.286.json`和`result_1.429.json`。

最后运行下面的命令，使用`wbf`将四个结果融合为一个结果。这个命令同样也可以用于**融合多个不同模型的结果**。

```bash
python wbf.py result.json result_1.143.json result_1.286.json result_1.429.json
```



最后，代码开源在: <https://github.com/misads/detection_template/tree/tile>。欢迎大家来star~~


