# pose_detection_openpose

## 现代 PyTorch 后端

本仓库现在提供 `pose_detection_openpose.extract_openpose_keypoints`：使用 YOLO 检测人物、使用
SDPose 推理 COCO17 关键点，并转换为 OpenPose COCO BODY18。它保留逐帧
`{name}_{frame:012d}_keypoints.json`、`people[*].pose_keypoints_2d`、最大人数、
起始帧和骨架视频接口，因此旧的 FCRN/VMD 工具可以直接读取，不再需要
OpenPose 可执行文件。

本程序是一个用于简化 [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 运行的批处理程序。

关于安装、启动选项、输出结果等，请查看上述 URL。

## 功能概要

输出供以下程序使用的 Openpose 数据。

 - [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd)
 - [miu200521358/3dpose_gan_vmd](https://github.com/miu200521358/3dpose_gan_vmd)
 - [miu200521358/FCRN-DepthPrediction-vmd](https://github.com/miu200521358/FCRN-DepthPrediction-vmd)
 - [miu200521358/VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi)


## 准备

详细信息请参阅 [Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)。

## 运行方法

### 视频的情况

1. 将简易启动批处理复制到 `Openpose` 的运行目录（`bin` 之类的上一级目录）
    - PortableDemo 版：复制到有 `LICENSE` 文件的层级，按版本复制对应的批处理
      - 1.3 … [PortableDemo/1.3/OpenposeVideo.bat](PortableDemo/1.3/OpenposeVideo.bat)
      - 1.4 … [PortableDemo/1.4/OpenposeVideo.bat](PortableDemo/1.4/OpenposeVideo.bat)
      - [OpenposeVideo_en.bat](OpenposeVideo_en.bat) 是英文版。!! 日志仍为日文。
    - 自行构建版：将 [self-build/OpenposeVideo.bat](self-build/OpenposeVideo.bat) 复制到 `x64` 目录下
    - [OpenposeVideo_en.bat](OpenposeVideo_en.bat) 是英文版。!! 日志仍为日文。
1. 运行 [OpenposeVideo.bat](OpenposeVideo.bat)
1. 程序会询问 `解析对象影像文件路径`，请输入视频文件的完整路径
1. 程序会询问 `影像中出现的最大人数`，请以 1 起始指定希望从影像中读取的最大人数
	- 未指定时，默认设为 1（单人解析）
1. 程序会询问 `解析起始帧 No`，请以 0 起始指定开始解析的帧编号
	- 当片头因 logo 等原因没有人物出现时，通过指定人物开始出现的第一帧编号，可以跳过开头的帧
	- 未指定时，默认设为 0（从第 0 帧开始解析）
1. 开始处理
1. 处理结束后，会在以下位置输出结果。
    - `解析对象影像文件路径/{解析对象影像文件名}_{执行日期时间}/{解析对象影像文件名}_json` 目录
        - → json 格式的 keypoints 数据
    - `解析对象影像文件路径/{解析对象影像文件名}_{执行日期时间}/{解析对象影像文件名}_openpose.avi`
        - → 在原影像上叠加 Openpose 解析结果的 avi 数据

### 图像的情况

1. 将简易启动批处理复制到 `Openpose` 的运行目录（`bin` 之类的上一级目录）
    - PortableDemo 版：复制到有 `LICENSE` 文件的层级，按版本复制对应的批处理
      - 1.3 … [PortableDemo/1.3/OpenposeImage.bat](PortableDemo/1.3/OpenposeImage.bat)
      - 1.4 … [PortableDemo/1.4/OpenposeImage.bat](PortableDemo/1.4/OpenposeImage.bat)
      - [OpenposeImage_en.bat](OpenposeImage_en.bat) 是英文版。!! 日志仍为日文。
    - 自行构建版：将 [self-build/OpenposeImage.bat](self-build/OpenposeImage.bat) 复制到 `x64` 目录下
    - [OpenposeImage_en.bat](OpenposeImage_en.bat) 是英文版。!! 日志仍为日文。
1. 运行 [OpenposeImage.bat](OpenposeImage.bat)
1. 程序会询问 `解析对象图像目录路径`，请输入存放图像的目录的完整路径
    - 目录内可放置多张图像
1. 程序会询问 `影像中出现的最大人数`，请以 1 起始指定希望从影像中读取的最大人数
	- 未指定时，默认设为 1（单人解析）
1. 开始处理
1. 处理结束后，会在以下位置输出结果。
    - `解析对象图像目录路径/{解析对象图像目录名}_{执行日期时间}/{解析对象图像目录名}_json` 目录/{解析对象图像文件名}_keypoints.json.png
        - → json 格式的 keypoints 数据
    - `解析对象图像目录路径/{解析对象图像目录名}_{执行日期时间}/{解析对象图像目录名}_openpose/{解析对象图像文件名}_rendered.png`
        - → 在原图像上叠加 Openpose 解析结果的 png 数据
1. ※图像解析结果无法用于 3d-pose-baseline-vmd 之后的流程。

## 注意事项

- `JSON 输出目标目录路径` 中请勿使用 12 位数字串。
    - 因为之后要从形如 `short02_000000000000_keypoints.json` 的 `{任意文件名}_{帧编号}_keypoints.json` 文件名中，将 12 位数字提取为帧编号

## 许可证
GNU GPLv3

公开、分发 MMD 自动追踪的结果时，请务必确认许可证并注明。Unity 等其他应用程序的情况亦同。

[MMD 动作追踪自动化套件许可证](https://ch.nicovideo.jp/miu200521358/blomaga/ar1686913)
