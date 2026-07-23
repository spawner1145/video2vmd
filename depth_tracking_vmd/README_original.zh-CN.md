# 基于全卷积残差网络的深度预测（Deeper Depth Prediction with Fully Convolutional Residual Networks）

作者：[Iro Laina](http://campar.in.tum.de/Main/IroLaina)、[Christian Rupprecht](http://campar.in.tum.de/Main/ChristianRupprecht)、[Vasileios Belagiannis](http://www.robots.ox.ac.uk/~vb/)、[Federico Tombari](http://campar.in.tum.de/Main/FedericoTombari)、[Nassir Navab](http://campar.in.tum.de/Main/NassirNavab)。

## 目录
0. [简介](#简介)
0. [快速指南](#快速指南)
0. [模型](#模型)
0. [结果](#结果)
0. [引用](#引用)
0. [许可证](#许可证)


## 简介

本仓库包含了论文「[Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373)」中所描述的、用于从单张 RGB 图像进行深度预测的已训练 CNN 模型。所提供的模型即为在论文中用于在基准数据集 NYU Depth v2（室内场景）和 Make3D（室外场景）上获得报告结果的模型。此外，所提供的代码还可用于对任意图像进行推理。


## 快速指南

已训练的模型目前以两种框架提供：MatConvNet 和 TensorFlow。有关如何上手的更多信息，请阅读以下内容。

### TensorFlow
*tensorflow* 文件夹中提供的代码相应地需要成功安装 [TensorFlow](https://www.tensorflow.org/) 库（任意平台）。
模型的计算图在 ```fcrn.py``` 中构建，对应的权重可通过下方链接下载。该实现基于 [ethereon 的](https://github.com/ethereon/caffe-tensorflow) Caffe-to-TensorFlow 转换工具。
```predict.py``` 提供了使用该网络预测输入图像深度图的示例代码。使用 ```python predict.py NYU_FCRN.ckpt yourimage.jpg``` 来试用此代码。

### MatConvNet

**前提条件**

*matlab* 文件夹中提供的代码需要用于 CNN 的 [MatConvNet 工具箱](http://www.vlfeat.org/matconvnet/)。要求成功编译版本等于或高于 1.0-beta20 的库（无论是否启用 GPU 支持均可）。
此外，用户应修改 `evaluateNYU.m` 和 `evaluateMake3D.m` 中的 ``` matconvnet_path = '../matconvnet-1.0-beta20' ```，使其指向库所存放的正确路径。

**使用方法**

要在 NYU 或 Make3D *测试集* 上获取预测的深度图并进行评估，用户只需分别运行 `evaluateNYU.m` 或 `evaluateMake3D.m` 即可。请注意，所有所需的数据和模型届时会自动下载（如果尚不存在），除了按需设置选项 `opts` 和 `netOpts` 之外，无需任何进一步的用户干预。请确保有足够的可用磁盘空间（最多 5 GB）。预测结果最终会保存到指定目录中的 .mat 文件里。

或者，也可以运行 `DepthMapPrediction.m`，以便在测试模式下手动使用已训练的模型来预测任意图像的深度图。

## 模型

这些模型是全卷积的，并且在上采样 CNN 层中同样运用了残差学习的思想。此处我们提供最快的变体，其中使用了特征图交错（interleaving）来进行上采样。为此，提供了一个自定义层 `+dagnn/Combine.m`。

已训练的模型——即论文中的 **ResNet-UpProj**——也可在此处下载：

- NYU Depth v2：[MatConvNet 模型](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.zip)、[TensorFlow 模型 (.npy)](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy)、[TensorFlow 模型 (.ckpt)](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip)
- Make3D：[MatConvNet 模型](http://campar.in.tum.de/files/rupprecht/depthpred/Make3D_ResNet-UpProj.zip)、TensorFlow 模型（即将推出）


## 结果

**新增！** NYU-Depth-v2 数据集验证集的预测结果也可在[此处](http://campar.in.tum.de/files/rupprecht/depthpred/predictions_NYUval.mat)下载（.mat）。

在下表中，我们报告了评估后应当获得的结果，并且与其他（最新的）单张图像深度预测方法进行了比较。
- NYU Depth v2 上的误差指标：

| NYU 上的最新技术水平        |  rel  |  rms  | log10 |
|-----------------------------|:-----:|:-----:|:-----:|
| [Roy & Todorovic](http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr16_NRF.pdf) (_CVPR 2016_) | 0.187 | 0.744 | 0.078 |
| [Eigen & Fergus](http://cs.nyu.edu/~deigen/dnl/) (_ICCV 2015_)  | 0.158 | 0.641 |   -   |
| **本文方法**                        | **0.127** | **0.573** | **0.055** |

- Make3D 上的误差指标：

| Make3D 上的最新技术水平     |  rel  |  rms  | log10 |
|-----------------------------|:-----:|:-----:|:-----:|
| [Liu et al.](https://bitbucket.org/fayao/dcnf-fcsp) (_CVPR 2015_)      | 0.314 |  8.60 | 0.119 |
| [Li et al.](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_Depth_and_Surface_2015_CVPR_paper.pdf) (_CVPR 2015_)      | 0.278 | 7.19 | 0.092 |
| **本文方法**                        | **0.175** |  **4.45** | **0.072** |

- 定性结果：
![Results](http://campar.in.tum.de/files/rupprecht/depthpred/images.jpg)

## 引用

如果你在研究中使用了此方法，请引用：

    @inproceedings{laina2016deeper,
            title={Deeper depth prediction with fully convolutional residual networks},
            author={Laina, Iro and Rupprecht, Christian and Belagiannis, Vasileios and Tombari, Federico and Navab, Nassir},
            booktitle={3D Vision (3DV), 2016 Fourth International Conference on},
            pages={239--248},
            year={2016},
            organization={IEEE}
    }

## 许可证

Simplified BSD License（简化版 BSD 许可证）

> 以下为许可证正式条款，以原始英文版本为法律效力依据，故保留原文。

Copyright (c) 2016, Iro Laina  
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
