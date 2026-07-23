# Lifting from the Deep
Denis Tome'、Chris Russell、Lourdes Agapito

[Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tome_Lifting_From_the_CVPR_2017_paper.pdf)，CVPR 2017

本项目基于 GNU GPLv3 许可证的条款授权。使用本软件即表示你同意该许可协议的条款（[链接](https://github.com/DenisTome/Lifting-from-the-Deep-release/blob/master/LICENSE)）。

![Teaser?](https://github.com/DenisTome/Lifting-from-the-Deep-release/blob/master/data/images/teaser-github.png)
## 摘要

我们针对从单张原始 RGB 图像进行 3D 人体姿态估计的问题，提出了一个统一的表述（formulation），它对 2D 关节估计与 3D 姿态重建进行联合推理，从而同时改善这两项任务。我们采用一种一体化的方法，将 3D 人体姿态的概率知识与多阶段 CNN 架构相融合，并利用对合理 3D 关键点位置的认知来优化对更佳 2D 位置的搜索。整个过程以端到端的方式训练，极为高效，并在 Human3.6M 上取得了最先进的结果，在 2D 和 3D 误差上均超越了以往的方法。

## 依赖关系

代码兼容 python2.7
- [Tensorflow 1.0](https://www.tensorflow.org/)
- [OpenCV](http://opencv.org/)

## 模型

在本演示中，**2D 姿态估计** 使用了在 MPI 数据集上训练的 CPM caffe 模型（[链接](https://github.com/shihenw/convolutional-pose-machines-release/tree/master/model)），而 **3D 姿态估计** 则使用了我们在 [Human3.6M 数据集](http://vision.imar.ro/human3.6m/description.php) 上训练的概率 3D 姿态模型。

## 测试
- 首先，运行 `setup.sh` 来获取已训练的模型并安装外部工具。
- 运行 `demo.py` 来对测试图像进行评估。

## 补充材料
- 项目[网页](http://visual.cs.ucl.ac.uk/pubs/liftingFromTheDeep/)
- 一些[视频](https://youtu.be/tKfkGttx0qs)。

## 引用

	@InProceedings{Tome_2017_CVPR,
	author = {Tome, Denis and Russell, Chris and Agapito, Lourdes},
	title = {Lifting From the Deep: Convolutional 3D Pose Estimation From a Single Image},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {July},
	year = {2017}
	}

## 说明

为本演示提供的模型 **并非** 用于生成论文结果的模型。我们仍在转换所有代码的过程中。

## 参考

- [Convolutional Pose Machines (CPM)](https://github.com/shihenw/convolutional-pose-machines-release)。
