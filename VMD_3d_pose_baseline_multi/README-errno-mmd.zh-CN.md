# VMD-Lifting

VMD-Lifting 是「Lifting from the Deep」的一个 fork，它会将估计出的 3D 姿态数据输出为 VMD 文件。

「Lifting from the Deep」的作者是 Denis Tome'、Chris Russell 和 Lourdes Agapito。
有关原始「Lifting from the Deep」的更多信息，请参阅「README-original.md」以及 http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/ 。

本项目基于 GNU GPLv3 许可证的条款授权。使用本软件即表示你同意该许可协议的条款（参见 LICENSE 文件）。

注意：用于头部姿态估计的默认训练模型「shape_predictor_68_face_landmarks.dat」是在 iBUG 300-W 面部关键点数据集上训练的。而 iBUG 300-W 数据集的许可证禁止商业用途。因此，如果你想在商业产品中使用此模型文件，应联系伦敦帝国理工学院（Imperial College London）以确认是否可行。

## 概要

这是一个从照片中推定人物姿态，并输出 VMD 格式的动作（姿态）数据的程序。
姿态推定使用了 Lifting from the Deep (https://github.com/DenisTome/Lifting-from-the-Deep-release) 的程序。

## 所需环境
- python (3.x 或 2.7)
- [Tensorflow](https://www.tensorflow.org/)
- [OpenCV](http://opencv.org/)
- python-tk (Tkinter)
- PyQt5
- dlib

在 Ubuntu 或 Debian GNU/Linux 环境下，切换到 root 后执行以下命令即可备齐所需环境。

```
# apt-get install python-pip
# pip install tensorflow-gpu
# apt-get install python-opencv
# apt-get install python-tk
# apt-get install python-pyqt5
# pip install dlib
```

在 Windows 环境下，请按以下步骤安装所需环境。

- 安装 cygwin：https://cygwin.com/install.html

- 按照 https://www.tensorflow.org/install/install_windows 的说明，安装 CUDA、cuDNN、Python 3.6

- 设置环境变量 PATH，使其使用上面安装的 python，而非 cygwin 的 python

- 用 pip 安装 tensorflow

`$ pip install  tensorflow-gpu`

- 安装 OpenCV

`$ pip install opencv-python`

- 安装 PyQt5

`$ pip install PyQt5`

- 安装 dlib

`$ pip install dlib`

## 准备
- 首先执行 setup.sh。此脚本会获取所需数据并安装外部工具。
- （接下来，如果想确认 Lifting from the Deep 本体的运行情况，可在 application 目录下执行 demo.py。）

- 如果要使用 dlib + OpenCV 进行 Head Pose Estimation（头部姿态推定），请下载 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 ，并将解压得到的文件 shape_predictor_68_face_landmarks.dat 放置到 applications/predictor/ 中。

```
$ mkdir applications/predictor
$ cd applications/predictor
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## 使用方法

cd application

./vmdlifting.py IMAGE_FILE VMD_FILE [POSITION_FILE]

- IMAGE_FILE：输入源图像文件（JPEG、PNG 等）
- VMD_FILE：输出目标 VMD 文件
- POSITION_FILE（可选）：输出关节位置的文本文件。用于调试。

使用示例：

./vmdlifting.py photo.jpg estimated.vmd

## 关于 Lifting from the Deep

Lifting from the Deep 是一种利用卷积神经网络（CNN），从单张 RGB 图像进行 3D 姿态推定的方法（的论文）及其实现程序。
作者是 Denis Tome'、Chris Russell 和 Lourdes Agapito。
详情请参阅项目网页（ http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/ ）中的论文和视频。

## 关于许可证
（如开头用英文所述）采用 GNU GPLv3 license。详情请阅读 LICENSE 文件。
另外，用于推定面部朝向的已训练模型 shape_predictor_68_face_landmarks.dat，
其训练所用的 iBUG 300-W 数据集不允许商业用途。
如需商业使用，请向 Imperial College London 取得许可，或另行准备其他已训练模型。

## 参考文献

D. Tome, C. Russell and L. Agapito. Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
