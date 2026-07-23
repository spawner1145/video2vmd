# 3d-pose-baseline-vmd

本程序由 miu(miu200521358) 基于 [3d-pose-baseline](https://github.com/ArashHosseini/3d-pose-baseline/)（ArashHosseini）fork 并改造而来。

关于运行的详细信息，请查看上述 URL，或参阅 [README-ArashHosseini.md](README-ArashHosseini.md)。

## 功能概要

- 从 [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 检测出的人体骨骼结构生成 3D 人体模型。
- 在生成 3D 人体模型时，输出关节数据
    - 通过在 [VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) 中读取关节数据，可生成 vmd（MMD 动作数据）文件
- 可解析多人的 OpenPose 数据。
    - ~~截至 2018/05/07，尚无法准确解析。请仅尝试单人解析。~~
    - 在 ver1.00(2019/02/13) 中已支持多人追踪。详情请查看 [FCRN-DepthPrediction-vmd](https://github.com/miu200521358/FCRN-DepthPrediction-vmd)。

## 准备

详细信息请参阅 [Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)。

### 依赖关系

请在 python3 系环境下安装以下内容：

* [h5py](http://www.h5py.org/)
* [tensorflow](https://www.tensorflow.org/) 1.0 或更高版本

### H36M 数据

3D 骨骼信息基于 [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 创建。
请从以下链接下载压缩文件，解压后放置到 `data` 目录下。

[H36M 数据 zip (Dropbox)](https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip)

### 学习数据

原始的学习数据会触及 Windows 的 260 字符路径限制，因此简化了路径后重新生成。
请从以下链接下载压缩文件，解压后放置到 `experiments` 目录下。

[学习数据 zip (GoogleDrive)](https://drive.google.com/file/d/1v7ccpms3ZR8ExWWwVfcSpjMsGscDYH7_/view?usp=sharing)

## 运行方法

1. 用 [Openpose 简易启动批处理](https://github.com/miu200521358/openpose-simple) 解析数据
1. 用 [深度推定](https://github.com/miu200521358/FCRN-DepthPrediction-vmd) 生成深度推定数据以及按人物索引区分的数据
1. 运行 [OpenposeTo3D.bat](OpenposeTo3D.bat)
	- [OpenposeTo3D_en.bat](OpenposeTo3D_en.bat) 是英文版。!! 日志仍为日文。
1. 程序会询问 `按 INDEX 区分的目录路径`，请指定第 2 步 `按人物索引区分的路径` 的完整路径
	- `{视频文件名}_json_{执行日期时间}_index{第 0 帧从左起的顺序}`
	- 多人追踪的情况下，需要分别运行
1. 程序会询问 `是否输出详细日志`，需要输出时请输入 `yes`
    - 未指定或为 `no` 时，输出普通日志（各参数文件与 3D 化动画 GIF）
    - 为 `warn` 时，不生成 3D 化动画 GIF（相应地更快）
    - 为 `yes` 时，输出详细日志，除日志消息外，还会输出调试用图像（相应地更慢）
1. 开始处理
1. 处理结束后，会在第 3 步的 `按人物索引区分的路径` 内输出以下结果。
    - pos.txt … 全帧的关节数据（[VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) 所需）详情：[Output](doc/Output.md)
    - start_frame.txt … 起始帧索引（[VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) 所需）
    - smoothed.txt … 全帧的 2D 位置数据（[VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) 所需）详情：[Output](doc/Output.md)
    - movie_smoothing.gif … 将各帧姿态拼接而成的动画 GIF
    - smooth_plot.png … 将移动量平滑化后的图表
    - frame3d/tmp_0000000000xx.png … 各帧的 3D 姿态
    - frame3d/tmp_0000000000xx_xxx.png … 各帧不同角度的 3D 姿态（仅在详细日志 yes 时）

## 注意事项

- Openpose 的 json 任意文件名中请勿使用 12 位数字串。
    - 因为要从形如 `short02_000000000000_keypoints.json` 的 `{任意文件名}_{帧编号}_keypoints.json` 文件名中，将 12 位数字提取为帧编号

## 许可证
MIT

公开、分发 MMD 自动追踪的结果时，请务必确认许可证并注明。Unity 等其他应用程序的情况亦同。

[MMD 动作追踪自动化套件许可证](https://ch.nicovideo.jp/miu200521358/blomaga/ar1686913)
