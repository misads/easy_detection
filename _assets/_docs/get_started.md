# 安装和使用教程

- [环境需求](#%E7%8E%AF%E5%A2%83%E9%9C%80%E6%B1%82)
- [训练和验证模型 voc或coco数据集](#%E8%AE%AD%E7%BB%83%E5%92%8C%E9%AA%8C%E8%AF%81%E6%A8%A1%E5%9E%8Bvoc%E6%88%96coco%E6%95%B0%E6%8D%AE%E9%9B%86)
  - [训练模型(单卡)](#3-%E5%8D%95%E5%8D%A1%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)
  - [训练模型(多卡)](#faster-rcnn-1)
  - [验证模型](#5-%E9%AA%8C%E8%AF%81%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)
- [自定义数据集](#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%A5voc%E6%A0%BC%E5%BC%8F%E4%B8%BA%E4%BE%8B)
- [自定义检测模型](#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A3%80%E6%B5%8B%E6%A8%A1%E5%9E%8B)
- [任务监控](#web%E7%AB%AF%E7%9B%91%E6%8E%A7%E5%92%8C%E7%9B%91%E6%8E%A7%E4%BB%BB%E5%8A%A1)



## 环境需求

```yaml
python>=3.6
numpy>=1.16.0
torch>=1.0  # effdet需要torch>=1.5，如果不使用effdet，在network/__init__.py下将其注释掉
tensorboardX>=1.6
utils-misc>=0.0.5
mscv>=0.0.3
matplotlib>=3.1.1
opencv-python==4.2.0.34  # opencv>=4.4版本需要编译，耗时较长，建议安装4.2版本
opencv-python-headless==4.2.0.34
albumentations>=0.5.1  # 需要opencv>=4.2
scikit-image>=0.17.2
easydict>=1.9
timm==0.1.30  # timm >= 0.2.0 不兼容 
typing_extensions==3.7.2
tqdm>=4.49.0
PyYAML>=5.3.1
Cython>=0.29.16
pycocotools>=2.0  # 需要Cython
omegaconf>=2.0.0  # effdet依赖
ipdb>=0.13.9  # 调试工具

```

　　使用`pip install`逐行安装即可。也可以安装好pytorch后使用`bash ./install.sh`一键安装依赖项。

## 训练和验证模型voc或coco数据集

### 1. 克隆此项目

```bash
git clone https://github.com/misads/easy_detection
cd easy_detection
```

### 2. 准备数据集（以voc为例）

1. 下载voc数据集，这里提供一个VOC0712的网盘下载链接：<https://pan.baidu.com/s/1AYao-vYtHbTRN-gQajfHCw>，密码7yyp。

2. 在项目目录下新建`datasets`目录：

   ```bash
   mkdir datasets
   ```

3. 将voc数据集的`VOC2007`或者`VOC2017`目录移动`datasets/voc`目录。（推荐使用软链接）

   ```bash
   ln -s <VOC的下载路径>/VOCdevkit/VOC2017 datasets/voc
   ```

4. 数据准备好后，数据的目录结构看起来应该是这样的：

   ```yml
   easy_detection
       └── datasets
             ├── voc           
             │    ├── Annotations
             │    ├── JPEGImages
             │    └── ImageSets/Main
             │            ├── train.txt
             │            └── test.txt
             └── <其他数据集>
   ```


5. 预览数据集标注

　　运行下面的指令来预览数据集标注： 

   ```bash
   python3 preview.py --config configs/faster_rcnn_voc.yml
   tensorboard --logdir logs/preview
   ```

　　效果如下：
　　<img alt="visualize" src="https://raw.githubusercontent.com/misads/easy_detection/master/_assets/_imgs/preview.png" style="zoom:50%;" />


### 3. 单卡训练模型

#### Faster RCNN

```bash
python3 train.py --config configs/faster_rcnn_voc.yml
```

【[训练日志](https://raw.githubusercontent.com/misads/easy_detection/master/_assets/_logs/frcnn_voc.txt)】

#### YOLOv2

```bash
python3 train.py --config configs/yolo2_voc.yml
```

【[训练日志](https://raw.githubusercontent.com/misads/easy_detection/master/_assets/_logs/yolo2_voc.txt)】

`darknet19_448.conv.23`是Yolo2在`ImageNet`上的预训练模型，可以在yolo官网下载。[[下载地址]](https://pjreddie.com/media/files/darknet19_448.conv.23)。

#### YOLOv3

```bash
python3 train.py --config configs/yolo3_voc.yml
```

`darknet53.conv.74`可以在yolo官网下载。[[下载地址]](https://pjreddie.com/media/files/darknet53.conv.74)。

### 参数说明

| 作用                        | 参数                       | 示例                         | 说明                                                         |
| --------------------------- | -------------------------- | ---------------------------- | ------------------------------------------------------------ |
| 指定训练标签                | `--tag`                    | `--tag yolo2_voc`            | 日志会保存在`logs/标签`目录下，模型会保存在`checkpoints/标签`目录下。 |
| 选择配置文件                    | `--config`                  | `--config configs/faster_rcnn_voc.yml`              | **必须明确给定**。                                           |
| 加载之前的模型/恢复训练     | `--load`       | `--load pretrained/yolo2.pt` | `--resume`配合`--load`使用，会恢复上次训练的`epoch`和优化器。 |
| 调试模式                    | `--debug`                  | `--debug`                    | 调试模式下只会训练几个batch就会开始验证。      |


### 4. 多卡训练模型

#### Faster RCNN

```bash
# 8卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./dist_train.sh 8 --config configs/faster_rcnn_voc_dist.yml
```

### 5. 验证训练模型


1. 新建`pretrained`文件夹：

   ```bash
   mkdir pretrained
   ```

2. 以Faster-RCNN为例，下载[[预训练模型]](https://github.com/misads/easy_detection#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)，并将其放在`pretrained`目录下：

   ```yml
   easy_detection
       └── pretrained
             └── 0_voc_FasterRCNN.pt
   ```

3. 运行以下命令来验证模型的`mAP`指标：

**Faster RCNN**

   ```bash
   python3 eval.py --config configs/faster_rcnn_voc.yml --load pretrained/0_voc_FasterRCNN.pt
   ```

**YOLOv2**

   ```bash
   python3 eval.py --config configs/yolo2_voc.yml --load pretrained/0_voc_Yolo2.pt
   ```


4. 如果需要使用`Tensorboard`可视化预测结果，可以在上面的命令最后加上`--vis`参数。然后运行`tensorboard --logdir results/cache`查看检测的可视化结果。如下图所示：

　　<img alt="visualize" src="https://raw.githubusercontent.com/misads/easy_detection/master/_assets/_imgs/vis.png" style="zoom:50%;" />


5. 使用其他的模型只需要修改`--config`参数使用不同配置文件即可。



## 自定义数据集(以voc格式为例)

### 1. 准备数据集

1. 将自己的数据集制作成`VOC`格式，并放在`datasets`目录下(可以使用软链接)。目录结构如下：

   ```yml
   easy_detection
       └── datasets
             └── mydata_dir    
                  ├── Annotations
                  ├── JPEGImages
                  └── ImageSets/Main
                          ├── train.txt
                          └── val.txt
   ```

2. 在`configs/data_roots`目录下新建一个`mydata.py`，内容如下：

   ```python
   class Data(object):
       data_format = 'VOC'
       voc_root = 'datasets/mydata_dir'
       train_split = 'train.txt'
       val_split = 'val.txt' 
       class_names = ["car", "person", "bus"]  # 所有的类别名
       
       img_format = 'jpg'  # 根据图片文件是jpg还是png设为'jpg'或者'png'
   ```

3. 在`configs`目录下复制**faster_rcnn_voc.yml**，改为**faster_rcnn_mydata.yml**，修改内容如下：


  ```yaml
  MODEL:
    NAME: Faster_RCNN
    BACKBONE: resnet50
  DATA:
    DATASET: mydata  # 这个注意要改一下
    TRANSFORM: frcnn
    SCALE: [600, 1000]
  OPTIMIZE:
    OPTIMIZER: sgd
    BASE_LR: 0.001 
    SCHEDULER: 1x
    BATCH_SIZE: 1
  ```

4. 完成定义数据集后，训练和验证时就可以使用`--config configs/faster_rcnn_mydata.yml`参数来使用自己的数据集。

### 2. 预览数据集标注

1. 运行`preview.py`：

   ```bash
   python3 preview.py --config faster_rcnn_mydata.yml
   ```

2. 运行`tensorboard`：

   ```bash
   tensorboard --logdir logs/preview
   ```

3. 打开浏览器，查看标注是否正确。

### 3. 在自定义数据集上训练已有模型

#### Faster RCNN

```bash
python3 train.py --config configs/faster_rcnn_mydata.yml
```

　　学习率和`batch_size`可以在config文件中视情况调整。

## 自定义检测模型

1. 复制`network`目录下的`Faster_RCNN`文件夹，改成另外一个名字(比如`MyNet`)。
2. 仿照`Faster_RCNN`的model.py，修改自己的网络结构、损失函数和优化过程。
3. 在`network/__init__.py`中`import`你模型的`Model`类并且在`models = {}`中添加它。

```python
from .Faster_RCNN.Model import Model as Faster_RCNN
from .MyNet.Model import Model as MyNet
models = {
    'Faster_RCNN': Faster_RCNN,
    'MyNet': MyNet,
}
```

4. 在config文件中将MODEL.NAME设置为自己的检测模型名称。

## WEB端监控和监控任务

1. 安装Flask

```bash
pip install Flask
```

2. 打开web服务

```bash
python serve.py -p8000 # 打开在8000端口
```

3. 用浏览器输入`localhost:8000`访问，效果如下：

<img alt="msboard" src="https://raw.githubusercontent.com/misads/easy_detection/master/_assets/_imgs/msboard.png" style="zoom:50%;" />
