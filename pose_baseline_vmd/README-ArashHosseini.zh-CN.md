## 3d-pose-baseline

这是以下论文的代码：

Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little.
_A simple yet effective baseline for 3d human pose estimation._
发表于 ICCV, 2017。https://arxiv.org/pdf/1705.03098.pdf 。

本仓库中的代码主要由以下人员编写：
[Julieta Martinez](https://github.com/una-dinosauria)、
[Rayat Hossain](https://github.com/rayat137) 和
[Javier Romero](https://github.com/libicocco)。

我们为 3D 人体姿态估计提供了一个强有力的基线（baseline），同时也揭示了当前方法所面临的挑战。我们的模型十分轻量，并且我们力求让代码保持透明、紧凑、易于理解。

### 依赖关系

* [h5py](http://www.h5py.org/)
* [tensorflow](https://www.tensorflow.org/) 1.0 或更高版本

### 首先
1. 观看我们的视频：https://youtu.be/Hmi3Pd9x1BE
2. 克隆本仓库并获取数据。我们提供了 [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 数据集的 3D 点、用于生成真值 2D 检测的相机参数，以及 [Stacked Hourglass](https://github.com/anewell/pose-hg-demo) 检测结果。

```bash
git clone https://github.com/una-dinosauria/3d-pose-baseline.git
cd 3d-pose-baseline
mkdir data
cd data
wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
unzip h36m.zip
rm h36m.zip
cd ..
```

### 快速演示

若要快速演示，你可以训练一个 epoch 并可视化结果。要开始训练，请运行

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 1`

在 GTX 1080 上，这大约需要不到 5 分钟完成，并在测试集上给出约 75 mm 的误差。

现在，要可视化结果，只需运行

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 1 --sample --load 24371`

这将生成类似如下的可视化效果：

![可视化示例](/imgs/viz_example.png?raw=1)


### 从 OpenPose 到 3d-Pose-Baseline

<p align="left">
    <img src="/imgs/open_pose_input.gif", width="360">
</p>

<p align="left">
    <img src="/imgs/output.gif", width="360">
</p>

1. 配置 [OpenPose](https://github.com/ArashHosseini/openpose)，并使用 `--write_json` 标志导出 Pose Keypoints（姿态关键点）。
2. 下载下方的预训练模型，然后直接运行

`python src/openpose_3dpose_sandbox.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --load 4874200 --openpose /path/to/openpose/output/json_directory --gif_fps 30` ，可选地也可加上 `--verbose 3` 用于调试，

通过中位数对曲线进行平滑

![掉帧](/imgs/dirty_plot.png?raw=1)

![平滑后的帧率](/imgs/smooth_plot.png?raw=1)


### 训练

要使用干净的 2D 检测结果训练模型，请运行：

<!-- `python src/predict_3dpose.py --camera_frame --residual` -->
`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise`

这对应于表 2 的最后一行。`Ours (GT detections) (MA)`

要基于 Stacked Hourglass 检测结果进行训练，请运行

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh`

这对应于表 2 的倒数第二行。`Ours (SH detections) (MA)`

在 GTX 1080 GPU 上，对于每批 64 个样本，前向+反向计算耗时 <8 ms，仅前向计算耗时 <6 ms。

### 预训练模型

我们还提供了一个基于 Stacked-Hourglass 检测结果的预训练模型，可通过 [google drive](https://drive.google.com/file/d/0BxWzojlLp259MF9qSFpiVjl0cU0/view?usp=sharing) 获取

要测试该模型，请在本项目的顶层目录解压该文件，然后调用

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --sample --load 4874200`

### 引用

如果你使用了我们的代码，请引用我们的工作

```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

### 许可证
MIT
