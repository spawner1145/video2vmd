# video2vmd

把视频中的人物动作转换为 MMD 可加载的 VMD 动作文件，并提供 2D 姿态、3D 骨架和 Blender MMD 视频预览。

当前主流程为：

```text
视频
  -> OpenPose BODY18 / SDPose COCO17
  -> BODY18 语义映射与时序平滑
  -> MotionAGFormer 2D -> 3D
  -> UniDepth 深度辅助
  -> 原版 VMD-3d-pose-baseline-multi 转换器
  -> full VMD
  -> Blender + mmd-tools MMD 视频
```

OpenPose 和 SDPose 都是 2D 姿态模型；MotionAGFormer 使用 Human3.6M 17 点，不是 COCO17，因此不能按数组编号直接拼接，代码中会先做语义映射。

## 目录

重要目录如下：

```text
3d_pose/
  process_dance.py                 主入口
  render_mmd_in_blender.py         Blender 渲染入口
  data/dance.mp4                   输入视频示例
  OpenPose_PyTorch/model/          OpenPose 权重
  SDPose_body/                     SDPose Body 模型目录
  unidepth-v2-vitl14/              UniDepth 模型目录
  motionagformer_b_h36m.pth        MotionAGFormer checkpoint
  VMD_3d_pose_baseline_multi/      原版 VMD 转换器
  pose_detection_openpose/         OpenPose/SDPose 后端
  pose_baseline_vmd/               MotionAGFormer 适配层
  depth_tracking_vmd/              UniDepth 适配层
  output/                          所有运行输出
```

## 环境

创建虚拟环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安装 PyTorch CUDA 版本(具体看你电脑cuda版本)：

```powershell
pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu130
```

安装项目依赖：

```powershell
pip install -r requirements.txt
```

确认 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU版本torch也可以运行，但 OpenPose、SDPose 和 MotionAGFormer 会明显变慢。

## 模型下载


以下模型都可以从 ModelScope 下载：

- SDPose Body：<https://www.modelscope.cn/models/Sunjian520/SDPose-Body>
- OpenPose PyTorch：<https://www.modelscope.cn/models/spawner/OpenPose_PyTorch>
- MotionAGFormer：<https://www.modelscope.cn/models/spawner/MotionAGFormer>
- UniDepth V2 ViT-L/14：<https://www.modelscope.cn/models/spawner/unidepth-v2-vitl14>

下载后建议保持以下目录和文件：

```text
SDPose_body/decoder/decoder.safetensors
SDPose_body/yolo11x.pt
OpenPose_PyTorch/model/body_pose_model.pth
unidepth-v2-vitl14/
motionagformer_b_h36m.pth[.tr后缀有没有无所谓]
```

`.pth.tr` 是常见的下载文件名后缀，PyTorch 可以直接读取。MotionAGFormer 的 checkpoint 必须和 `--motionagformer-size` 匹配。自定义路径示例：

```powershell
python process_dance.py data\dance.mp4 `
  --sdpose-model-dir D:\models\SDPose-Body `
  --openpose-model-dir D:\models\OpenPose_PyTorch\model `
  --unidepth-model-path D:\models\unidepth-v2-vitl14 `
  --motionagformer-size l `
  --motionagformer-checkpoint D:\models\motionagformer_l_h36m.pth.tr
```

## 模型目录

默认路径如下，所有路径都可以用命令行覆盖：

| 参数 | 默认路径 | 内容 |
|---|---|---|
| `--openpose-model-dir` | `OpenPose_PyTorch/model` | `body_pose_model.pth` |
| `--sdpose-model-dir` | `SDPose_body` | `unet/`、`vae/`、`decoder/`、`yolo11x.pt` 等 |
| `--unidepth-model-path` | `unidepth-v2-vitl14` | UniDepth V2 本地模型 |
| `--motionagformer-checkpoint` | `motionagformer_b_h36m.pth` | MotionAGFormer H36M checkpoint |

示例：

```powershell
python process_dance.py data\dance.mp4 `
  --openpose-model-dir D:\models\openpose\model `
  --sdpose-model-dir D:\models\SDPose_body `
  --unidepth-model-path D:\models\unidepth-v2-vitl14 `
  --motionagformer-checkpoint D:\models\motionagformer_b_h36m.pth
```

默认使用纯 SDPose。如果 SDPose 目录或 `decoder/decoder.safetensors` 不存在，会自动回退到 OpenPose。

## 最简单的完整流程

使用默认模型目录和默认输出目录：

```powershell
python process_dance.py data\dance.mp4 --stage all
```

默认使用纯 SDPose COCO17，并转换为 BODY18 后进入后续流程。默认 VMD 为 full 模式，位置和旋转阈值都是 `0`，不会删除关键帧。需要混合 OpenPose 和 SDPose 时显式传入 `--pose-source adaptive`。

主要输出：

```text
output/dance_pipeline/
  openpose_body18.npz
  pose_data.npz
  pose_2d_preview.mp4
  pose_3d_preview.mp4
  pose_3d_fused_preview.mp4
  dance_motion_full.vmd
  fused_vmd_inputs/dance_motion_fused_full.vmd
  run_summary.json
```

## 四组推荐实验

每组使用独立目录，方便比较。

### OpenPose，头部按姿态估计

```powershell
python process_dance.py data\dance.mp4 `
  --stage all `
  --output-dir output\openpose_natural `
  --pose-source openpose
```

### OpenPose，头部始终面向镜头

```powershell
python process_dance.py data\dance.mp4 `
  --stage all `
  --output-dir output\openpose_camera `
  --pose-source openpose `
  --head-facing-camera
```

### 纯 SDPose，头部按姿态估计

当前参数名保留为 `sdpose-limbs` 以兼容已有命令，但它现在表示纯 SDPose COCO17 转 BODY18：

```powershell
python process_dance.py data\dance.mp4 `
  --stage all `
  --output-dir output\sdpose_natural `
  --pose-source sdpose-limbs
```

### 纯 SDPose，头部始终面向镜头

```powershell
python process_dance.py data\dance.mp4 `
  --stage all `
  --output-dir output\sdpose_camera `
  --pose-source sdpose-limbs `
  --head-facing-camera
```

该命令的输出效果如下：

**2D 姿态预览**（`pose_2d_preview.mp4`，SDPose COCO17 转 BODY18）：

<video src="output/sdpose_camera/pose_2d_preview.mp4" controls muted width="100%"></video>

**Blender MMD 渲染结果**（`maid_aris_sdpose_camera.mp4`，头部始终面向镜头）：

<video src="output/sdpose_camera/maid_aris_sdpose_camera.mp4" controls muted width="100%"></video>


示例中的 MMD 模型「女仆爱丽丝（maid_aris）」来自 [冬瓜炒菜](https://www.aplaybox.com/details/model/KwWftXgt8B8P)，使用和发布请遵守原作者的使用条款。

## 分阶段运行

完整流程较慢时，可以分阶段运行：

```powershell
# 只检测 2D 姿态
python process_dance.py data\dance.mp4 --stage pose --output-dir output\test

# 只运行 UniDepth 和人物跟踪
python process_dance.py data\dance.mp4 --stage depth --output-dir output\test

# 只运行 MotionAGFormer
python process_dance.py data\dance.mp4 --stage lift --output-dir output\test

# 使用已有缓存导出 VMD 和预览
python process_dance.py data\dance.mp4 --stage export --output-dir output\test
```

检测或模型更换后需要强制重算：

```powershell
python process_dance.py data\dance.mp4 --stage all --output-dir output\test --force
```

## VMD 选项

### 保留每一帧

默认值已经是：

```text
--vmd-position-threshold 0
--vmd-rotation-threshold 0
```

因此默认输出 `dance_motion_full.vmd`，不会做关键帧约简。不要把阈值设为正数，除非你明确需要减小 VMD 文件体积。

### 固定头部面向镜头

```powershell
--head-facing-camera
```

该选项会抵消上半身父骨骼对“首”的影响，并将“頭”保持中性旋转。身体仍然使用正常姿态旋转。

### 翻转指定点的前后方向

BODY18 编号：

```text
0 Nose       1 Neck       2 RShoulder  3 RElbow   4 RWrist
5 LShoulder  6 LElbow     7 LWrist     8 RHip     9 RKnee
10 RAnkle    11 LHip      12 LKnee     13 LAnkle  14 REye
15 LEye      16 REar      17 LEar
```

例如翻转两个髋点：

```powershell
--flip-depth-joints 8,11
```

这只是实验选项，默认不翻转任何点。

## Blender MMD 渲染

我用的是 Blender 3.6，需要安装 `mmd-tools` 插件。使用 PMX 和 VMD 渲染：

```powershell
& "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe" `
  --background `
  --python render_mmd_in_blender.py -- `
  --pmx "D:\code\epicmmd\maid_aris\maidaris.pmx" `
  --vmd "D:\code\3d_pose\output\openpose_natural\fused_vmd_inputs\dance_motion_fused_full.vmd" `
  --output "D:\code\3d_pose\output\openpose_natural\maid_aris_openpose_natural.mp4"
```

渲染脚本会固定渲染帧 `0..299`，因此 300 帧输入对应 300 帧输出，不会丢掉开头动作，也不会额外增加末帧。

## 输出文件说明

- `openpose_body18.npz`：BODY18 2D 关键点和置信度。
- `sdpose_coco17.npz`：SDPose 原始 COCO17 缓存。
- `motionagformer_pose.npz`：MotionAGFormer H36M17 输出。
- `unidepth_metric.npz`：UniDepth 逐帧深度。
- `pose_data.npz`：最终 2D、3D、融合深度和置信度。
- `pose_2d_preview.mp4`：2D BODY18 预览。
- `pose_3d_preview.mp4`：MotionAGFormer 3D 预览。
- `pose_3d_fused_preview.mp4`：深度约束后的 3D 预览。
- `dance_motion_full.vmd`：原版姿态转换输出。
- `fused_vmd_inputs/dance_motion_fused_full.vmd`：融合深度版本。
- `run_summary.json`：模型、参数、帧数、防抖和翻转配置记录。

### 头部抖动

目前找不到一个好方法来解决这个问题，可以先使用：

```powershell
--head-facing-camera
```

该模式不使用 VMD 的头部姿态旋转。如果需要保留头部动作，应检查 `pose_3d_fused_preview.mp4`，并适当降低深度观测权重或使用防抖后的缓存重新导出。

### CUDA 显存不足

可以降低 UniDepth 分辨率：

```powershell
--unidepth-resolution-level 1
```

或者减少测试帧数：

```powershell
--max-frames 60
```

## 许可证和原始项目

本项目修改使用了 OpenPose PyTorch、SDPose-OOD、MotionAGFormer、UniDepth 以及 `miu200521358/3d-pose-baseline-vmd` / `miu200521358/VMD-3d-pose-baseline-multi`。使用和发布模型、权重或生成结果时，请分别遵守各上游项目的许可证和模型使用条款。
