# cv_template


一个图像复原或分割的Baseline。

## To do List

- 平台支持
  - [ ] 多GPU支持
  - [ ] 测试时支持多`batch_size`
  
- TTA
  - [ ] 放大、色相、饱和度、亮度
  - [ ] `flip`
  - [ ] 多尺度测试
  - [ ] ttach库
  
- 其他Tricks
  - [ ] 使用fp_16训练，提高训练速度
  - [ ] One_Cycle 学习率

## Prerequisites

```yaml
python >= 3.6
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
mscv >= 0.0.3
```

## Code Usage

```python
python help.py
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