# depth_tracking_vmd

## 现代 PyTorch 后端

`depth_tracking_vmd.estimate_depth_and_track_people` 使用 UniDepth V2 代替 TensorFlow FCRN，
但继续调用原仓库的 `tensorflow/sort_people.py`。因此保留 `depth.txt`、
`depth_z.txt`、`idxXX/json`、多人身份排序、缺失点继承以及
`reverse_specific`/`order_specific` 人工纠错接口。统一入口为项目根目录的
`process_dance.py`。

本程序由 miu(miu200521358) 基于 [FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction)（Iro Laina 等人）fork 并改造而来。

关于运行的详细信息，请查看上述 URL，或参阅 [README-original.md](README-original.md)。

## 功能概要

- 从 [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 检测出的人体骨骼结构中推定深度。
- 基于 [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 的关节 XY 位置信息与深度推定结果，对多人的追踪进行人物跟踪。

## 准备

详细信息请参阅 [Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)。

### 依赖关系

请在 python3 系环境下安装以下内容：

- [OpenCV](http://opencv.org/)
- [tensorflow](https://www.tensorflow.org/) 1.0 ～ 1.6
- numpy
- python-dateutil
- pytz
- pyparsing
- six
- matplotlib
- opencv-python
- imageio

补充）如果是能够运行以下程序的环境，则无需额外安装。
 - [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd)
 - [miu200521358/VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi)

## 模型数据

创建「`tensorflow/data`」目录，并将 [tensorflow 用模型数据](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip) 放置在该目录下。

## 运行方法

1. 用 [Openpose 简易启动批处理](https://github.com/miu200521358/openpose-simple) 解析数据
1. 运行 [VideoToDepth.bat](VideoToDepth.bat)
	- [VideoToDepth_en.bat](VideoToDepth_en.bat) 是英文版。!! 日志仍为日文。
1. 程序会询问 `解析对象影像文件路径`，请输入视频文件的完整路径
1. 程序会询问 `解析结果 JSON 目录路径`，请指定第 1 步的结果目录路径
	- `{视频路径}/{视频文件名}_{执行年月日}/{视频文件名}_json` 即为对象目录路径
1. 程序会询问 `深度推定间隔`，请指定进行深度推定的帧间隔（仅限整数）
    - 每隔指定的间隔进行一次深度推定
    - 未指定时，默认为「10」
    - 值越小，深度推定越精细，但处理相应地越慢
1. 程序会询问 `翻转帧列表`，请指定 Openpose 误识别了正反面的帧范围。
	- 仅在此处指定的帧范围内进行翻转判定。
	- 可以像 `10,20` 这样用逗号分隔来指定多个帧。
	- 像 `10-15` 这样用连字符分隔时，可指定该范围内的帧。
1. 程序会询问 `顺序指定列表`，当交叉后人物跟踪出错时，用于指定帧编号与人物索引的顺序。
	- 人物索引以第 0 帧从左起数第 0 个、第 1 个……来计数。
	- 指定 `[12:1,0]` 时，第 12 帧从画面左侧起，按第 0 帧的第 1 个、第 0 帧的第 0 个重新排列。
	- 指定 `[12-15:1,0]` 时，在第 12～15 帧范围内，按第 1 个、第 0 个重新排列。
1. 程序会询问 `是否输出详细日志`，需要输出时请输入 `yes`
    - 未指定或为 `no` 时，输出普通日志
1. 开始处理
1. 处理结束后，会在与 `解析结果 JSON 目录路径` 同级的位置输出以下结果
	- `{视频文件名}_json_{执行日期时间}_depth`
	    - depth.txt … 各关节位置的深度推定值列表
	    - message.log … 输出顺序等参数指定信息的输出日志
	    - movie_depth.gif … 深度推定的合成动画 GIF
	        - 白色的点为作为关节位置获取到的点
	    - depth/depth_0000000000xx.png … 各帧的深度推定结果
	    - ※进行多人追踪时，会输出所有人的深度信息
	- `{视频文件名}_json_{执行日期时间}_index{第 0 帧从左起的顺序}`
	    - depth.txt … 该人物各关节位置的深度推定值列表
1. message.log 中输出的信息
	- ＊＊第 05254 帧的输出顺序: [5254:1,0], 位置: {0: [552.915, 259.182], 1: [654.837, 268.902]}
		- 在第 5254 帧，按 1, 0 的顺序进行了分配
			- 设为第 0 个的 1，推定为 [654.837, 268.902] 的人物
			- 设为第 1 个的 0，推定为 [552.915, 259.182] 的人物
		- 如果这一帧的人物站位有误，可将 [5254:0,1] 指定到 `顺序指定列表` 中，第 5254 帧的输出顺序即会翻转
	- ※※第 03329 帧 有顺序指定 [1, 0]
		- 第 3229 帧在 `顺序指定列表` 中被指定为 [1,0]，并据此输出
	- ※※第 04220 帧 第 1 个人物、下半身翻转 [4220:1]
		- 第 4220 帧在 `翻转帧列表` 中被指定，且被判定为翻转时，进行了翻转输出


## 许可证
Simplified BSD License

公开、分发 MMD 自动追踪的结果时，请务必确认许可证并注明。Unity 等其他应用程序的情况亦同。

[MMD 动作追踪自动化套件许可证](https://ch.nicovideo.jp/miu200521358/blomaga/ar1686913)
