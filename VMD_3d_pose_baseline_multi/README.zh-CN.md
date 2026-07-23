# VMD-3d-pose-baseline-multi

本程序由 miu(miu200521358) 基于 [VMD-Lifting](https://github.com/errno-mmd/VMD-Lifting)（errno-mmd）fork 并改造而来。

关于运行的详细信息，请查看上述 URL，或参阅 [README-errno-mmd.md](README-errno-mmd.md) 以及 [README-original.md](README-original.md)。

## 功能概要

从以下程序生成的 3D 关节及其他数据中，生成 vmd（MMD 动作数据）文件

 - [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd)
 - [miu200521358/3dpose_gan_vmd](https://github.com/miu200521358/3dpose_gan_vmd)
 - [miu200521358/FCRN-DepthPrediction-vmd](https://github.com/miu200521358/FCRN-DepthPrediction-vmd)
 - [miu200521358/VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi)


## 准备

详细信息请参阅 [Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)。

### 依赖关系

请在 python3 系环境下安装以下内容：

- [Tensorflow](https://www.tensorflow.org/)
- [OpenCV](http://opencv.org/)
- python-tk (Tkinter)
- PyQt5

## 运行方法

1. 用 [Openpose 简易启动批处理](https://github.com/miu200521358/openpose-simple) 解析数据
1. 运行 [OpenposeTo3D.bat](OpenposeTo3D.bat)
	- [OpenposeTo3D_en.bat](OpenposeTo3D_en.bat) 是英文版。!! 日志仍为日文。
1. 准备好由 [miu200521358/FCRN-DepthPrediction-vmd](https://github.com/miu200521358/FCRN-DepthPrediction-vmd) 生成的深度推定数据（depth.txt）
1. 准备好由 [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd) 生成的 3D 关节数据（pos.txt）、2D 关节数据（smoothed.txt）
1. 准备好由 [miu200521358/3dpose_gan_vmd](https://github.com/miu200521358/3dpose_gan_vmd) 生成的 3D 关节数据（pos_gan.txt）、2D 关节数据（smoothed_gan.txt）
1. 运行 [3DToVmd.bat](3DToVmd.bat)
	- [3DToVmd_en.bat](3DToVmd_en.bat) 是英文版。!! 日志仍为日文。
1. 程序会询问 `3D 解析结果目录路径`，请指定第 1 步的结果目录路径
1. 程序会询问 `骨骼结构 CSV 文件`，请指定追踪目标模型的骨骼结构 CSV 文件路径
    - 计算中心（センター）移动时会用到身高
    - 骨骼结构 CSV 文件的输出方法，请参阅 [born/README.md](born/README.md)
    - 未指定时，默认读取 [born/あにまさ式ミクボーン.csv](born/あにまさ式ミクボーン.csv)
1. 程序会询问 `是否用 IK 输出足部`，需要时请输入 `yes`
    - 未指定或为 `yes` 时，用 IK 输出
    - 为 `no` 时，用 FK 输出
1. 程序会询问 `脚跟位置修正`，请指定脚跟的 Y 轴修正值（可为小数）
    - 输入负值会靠近地面，输入正值会远离地面。
    - 程序会在一定程度上自动修正，但当因高跟鞋或厚底鞋等无法完全修正时可进行设置
    - 未指定时，为「0」，不进行修正
1. 程序会询问 `中心 Z 移动倍率`，请输入中心（センター）Z 移动时的移动幅度
    - 值越小，Z 的移动量越少；值越大，移动量越多
    - 未指定时，默认为「5」
    - 指定为「0」时，不进行中心 Z 移动
1. 程序会询问 `平滑化次数`，请指定一个合适的正整数
    - 未指定时，默认指定 1 次
1. 程序会询问 `移动键抽稀量`，请指定移动键（中心・IK）抽稀时的移动量（可为小数）
    - 抽稀掉指定移动量范围内的移动关键帧
    - 未指定时，默认为「0.5」
    - 指定为「0」时，不进行抽稀，并跳过以下的 `旋转键抽稀角度`
1. 程序会询问 `旋转键抽稀角度`，请指定旋转类骨骼抽稀时的旋转角度（仅限 0～180 度的整数）
    - 若旋转在指定的角度范围内，则抽稀该关键帧
    - 未指定时，默认为「3」
1. 程序会询问 `是否输出详细日志`，需要输出时请输入 `yes`
    - 未指定或为 `no` 时，输出普通日志
1. 开始处理
1. 处理结束后，会在第 1 步的结果目录下输出 vmd 文件
	- output_{日期}_{时间}_u{直立帧 IDX}_h{脚跟位置修正}_z{中心 Z 移动倍率}_s{平滑化次数}_p{移动键抽稀量}_r{旋转键抽稀角度}_full/reduce.vmd
		- 未进行关键帧抽稀时，末尾为「full」。进行了抽稀时，为「reduce」。
	- upright.txt … 直立帧的关键帧信息
1. 启动 MMD，加载模型后，再加载动作

## 许可证
GNU GPLv3

公开、分发 MMD 自动追踪的结果时，请务必确认许可证并注明。Unity 等其他应用程序的情况亦同。

[MMD 动作追踪自动化套件许可证](https://ch.nicovideo.jp/miu200521358/blomaga/ar1686913)
