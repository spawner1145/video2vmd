"""MotionAGFormer backend for the original OpenPose-to-pos.txt pipeline."""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
H36M_EDGES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
              (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
              (12, 13), (8, 14), (14, 15), (15, 16))
ORIGINAL_INDICES = (0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27)
# MotionAGFormer uses left leg first and right arm first. The original VMD
# pipeline uses right leg first and left arm first.
MOTIONAG_TO_ORIGINAL = (0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13)
ORIGINAL_TO_MOTIONAG = tuple(np.argsort(MOTIONAG_TO_ORIGINAL))


def _read_body18(json_dir):
    files = sorted(Path(json_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No OpenPose JSON files found in {json_dir}")
    frames, points, confidence = [], [], []
    previous = None
    for item in files:
        frame = int(re.findall(r"(\d{12})", item.name)[0])
        people = json.loads(item.read_text(encoding="utf-8")).get("people", [])
        if people:
            values = np.asarray(people[0]["pose_keypoints_2d"], np.float32).reshape(18, 3)
            previous = values.copy()
        elif previous is not None:
            values = previous.copy(); values[:, 2] = 0
        else:
            values = np.zeros((18, 3), np.float32)
        frames.append(frame); points.append(values[:, :2]); confidence.append(values[:, 2])
    return np.asarray(frames), np.asarray(points), np.asarray(confidence)


def _interpolate(values):
    result = values.copy(); axis = np.arange(len(values))
    for joint in range(values.shape[1]):
        for dimension in range(values.shape[2]):
            good = np.isfinite(values[:, joint, dimension]) & (values[:, joint, dimension] != 0)
            if good.any():
                result[:, joint, dimension] = np.interp(axis, axis[good], values[good, joint, dimension])
    return result


def _smooth_body18(points, confidence):
    raw = _interpolate(points)
    med = median_filter(raw, size=(5, 1, 1), mode="nearest")
    smooth = savgol_filter(med, 7, 2, axis=0, mode="interp") if len(raw) >= 7 else med
    weight = np.clip((confidence - .35) / .35, 0, 1)[..., None]
    return (weight * raw + (1 - weight) * smooth).astype(np.float32)


def _read_original_smoothed_body18(json_dir):
    module_path = ROOT / "pose_baseline_vmd" / "src" / "openpose_utils.py"
    spec = importlib.util.spec_from_file_location("pose_baseline_openpose_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start_frame, smoothed = module.read_openpose_json(str(json_dir), 0, False)
    return start_frame, np.asarray([smoothed[key] for key in sorted(smoothed)], np.float32).reshape(-1, 18, 2)


def body18_to_h36m(body, confidence=None):
    result = np.zeros((len(body), 17, 3), np.float32)
    result[:, 0, :2] = (body[:, 8] + body[:, 11]) / 2
    result[:, 1:4, :2] = body[:, [11, 12, 13]]
    result[:, 4:7, :2] = body[:, [8, 9, 10]]
    result[:, 8, :2] = body[:, 1]
    result[:, 7, :2] = (result[:, 0, :2] + result[:, 8, :2]) / 2
    result[:, 9, :2] = (body[:, 1] + body[:, 0]) / 2
    result[:, 10, :2] = body[:, 0]
    result[:, 11:14, :2] = body[:, [2, 3, 4]]
    result[:, 14:17, :2] = body[:, [5, 6, 7]]
    if confidence is not None:
        result[:, :, 2] = np.clip(confidence.mean(1)[:, None], 0, 1)
    return result


def _flip(value):
    result = value.clone(); result[..., 0] *= -1
    left, right = [1, 2, 3, 14, 15, 16], [4, 5, 6, 11, 12, 13]
    original = result.clone(); result[:, :, left] = original[:, :, right]; result[:, :, right] = original[:, :, left]
    return result


def _load_model(checkpoint):
    sys.path.insert(0, str(ROOT / "MotionAGFormer"))
    from model.MotionAGFormer import MotionAGFormer
    model = MotionAGFormer(
        n_layers=16, dim_in=3, dim_feat=128, dim_rep=512, dim_out=3,
        mlp_ratio=4, num_heads=8, use_layer_scale=True,
        layer_scale_init_value=1e-5, use_adaptive_fusion=True,
        use_temporal_similarity=True, neighbour_num=2, n_frames=243).cuda().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    return model


def _lift(model, normalized):
    count = len(normalized); padding = max(0, 243 - count)
    padded = np.pad(normalized, ((0, padding), (0, 0), (0, 0)), mode="edge")
    starts = list(range(0, max(1, len(padded) - 242), 121)); last = len(padded) - 243
    if starts[-1] != last:
        starts.append(last)
    accumulated = np.zeros((len(padded), 17, 3), np.float32)
    weights = np.zeros((len(padded), 1, 1), np.float32)
    window = np.hanning(245)[1:-1].astype(np.float32)[:, None, None] + .05
    with torch.inference_mode():
        for start in tqdm(starts, desc="Lifting 2D keypoints with MotionAGFormer"):
            value = torch.from_numpy(padded[start:start + 243]).unsqueeze(0).cuda()
            prediction = (model(value) + _flip(model(_flip(value)))) / 2
            prediction = prediction[0].cpu().numpy(); prediction[:, 0] = 0
            accumulated[start:start + 243] += prediction * window
            weights[start:start + 243] += window
    return median_filter((accumulated / weights)[:count], size=(5, 1, 1), mode="nearest")


def _moving_average(values, width=91):
    result = np.empty(len(values), np.float32)
    half = width // 2
    for frame in range(len(values)):
        result[frame] = np.mean(values[max(0, frame - half):min(len(values), frame + half + 1)])
    return result


def _original_contract_pose(raw_h36m17, body18, video_width, video_height):
    raw = raw_h36m17[:, MOTIONAG_TO_ORIGINAL].astype(np.float32)
    combined_leg_length = (
        np.linalg.norm(raw[:, 2] - raw[:, 1], axis=1)
        + np.linalg.norm(raw[:, 3] - raw[:, 2], axis=1)
        + np.linalg.norm(raw[:, 5] - raw[:, 4], axis=1)
        + np.linalg.norm(raw[:, 6] - raw[:, 5], axis=1)
    )
    # The original VMD converter and its 20/100 mm stabilizing offsets assume
    # Human3.6M millimetres and a mean combined leg length of 1743 mm.
    raw *= 1743.0 / max(float(np.mean(combined_leg_length)), 1e-6)
    observed = np.zeros((len(body18), 17, 2), np.float32)
    observed[:, 0] = (body18[:, 8] + body18[:, 11]) / 2
    observed[:, 1:4] = body18[:, [8, 9, 10]]
    observed[:, 4:7] = body18[:, [11, 12, 13]]
    observed[:, 8] = 1.1 * body18[:, 1] - 0.1 * observed[:, 0]
    observed[:, 7] = (observed[:, 0] + observed[:, 8]) / 2
    observed[:, 9] = body18[:, 0]
    observed[:, 10] = (body18[:, 16] + body18[:, 17]) / 2
    observed[:, 11:14] = body18[:, [5, 6, 7]]
    observed[:, 14:17] = body18[:, [2, 3, 4]]

    length_2d = np.stack([
        np.linalg.norm(observed[:, end] - observed[:, start], axis=1)
        for start, end in H36M_EDGES], axis=1).sum(axis=1)
    length_3d = np.stack([
        np.linalg.norm(raw[:, end] - raw[:, start], axis=1)
        for start, end in H36M_EDGES], axis=1).sum(axis=1)
    xy_scale = _moving_average(length_3d) / np.maximum(_moving_average(length_2d), 1e-6)
    center_x, center_y = video_width / 2, video_height / 2
    camera_distance = 4000.0
    teacher_tilt = math.tan(math.radians(13.0))
    reconstructed = np.zeros_like(raw)

    for frame in range(len(raw)):
        for joint in range(17):
            dy = raw[frame, joint, 1] - raw[frame, 0, 1]
            dz = raw[frame, joint, 2] - raw[frame, 0, 2] - dy * teacher_tilt
            z_ratio = (camera_distance + dz) / camera_distance
            reconstructed[frame, joint, 0] = (observed[frame, joint, 0] - center_x) * xy_scale[frame] * z_ratio
            reconstructed[frame, joint, 1] = (observed[frame, joint, 1] - center_y) * xy_scale[frame] * z_ratio
            reconstructed[frame, joint, 2] = dz

        for joint in (9, 10):
            delta = raw[frame, joint] - raw[frame, 8]
            dz = delta[2] - delta[1] * teacher_tilt
            reconstructed[frame, joint, 2] = reconstructed[frame, 8, 2] + dz

        delta = raw[frame, 7] - raw[frame, 8]
        dz = delta[2] - delta[1] * teacher_tilt
        reconstructed[frame, 7] = reconstructed[frame, 8] + np.asarray([delta[0], delta[1], dz])

    lowest_foot = reconstructed[:, 1:7, 1].max(axis=1)
    for frame in range(len(reconstructed)):
        ground = np.median(lowest_foot[max(0, frame - 60):min(len(reconstructed), frame + 60)])
        reconstructed[frame, :, 1] -= ground

    # Original baseline writes X, depth, up after swapping Y/Z. Internally we
    # retain the converter-facing X/up/depth convention.
    converter_pose = reconstructed[:, :, [0, 1, 2]].copy()
    converter_pose[:, :, 1] = -reconstructed[:, :, 1]
    converter_pose[:, :, 2] = reconstructed[:, :, 2]
    motionag_order = converter_pose[:, ORIGINAL_TO_MOTIONAG]
    pixels_per_unit = float(1.0 / np.median(xy_scale))
    bone_lengths = np.stack([
        np.linalg.norm(motionag_order[:, end] - motionag_order[:, start], axis=1)
        for start, end in H36M_EDGES], axis=1).mean(axis=0)
    return converter_pose, motionag_order, pixels_per_unit, bone_lengths


def _write_original_files(output_dir, original_pose, body18, start_frame):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pos.txt").open("w", encoding="ascii") as handle:
        for frame_pose in original_pose:
            handle.write(", ".join(
                f"{index} {joint[0]:.8f} {joint[2]:.8f} {joint[1]:.8f}"
                for index, joint in zip(ORIGINAL_INDICES, frame_pose)) + ", \n")
    np.savetxt(output_dir / "smoothed.txt", body18.reshape(len(body18), 36), fmt="%.8f")
    (output_dir / "start_frame.txt").write_text(str(int(start_frame)), encoding="ascii")


def lift_openpose_sequence(json_dir, output_dir, *, video_width, video_height,
                       checkpoint=None, force=False):
    """Lift identity-sorted BODY18 JSON and emit original pipeline files."""
    json_dir, output_dir = Path(json_dir), Path(output_dir)
    checkpoint = Path(checkpoint or ROOT / "motionagformer_b_h36m.pth")
    artifact = output_dir / "motionagformer_pose.npz"
    if artifact.exists() and not force:
        return artifact
    frames, body18, confidence = _read_body18(json_dir)
    start_frame, smoothed = _read_original_smoothed_body18(json_dir)
    inputs = body18_to_h36m(smoothed, confidence)
    inputs[:, :, :2] = inputs[:, :, :2] / video_width * 2 - np.array([1, video_height / video_width], np.float32)
    model = _load_model(checkpoint)
    raw = _lift(model, inputs)
    original_pose, pose, pixels_per_unit, bone_lengths = _original_contract_pose(
        raw, smoothed, video_width, video_height)
    _write_original_files(output_dir, original_pose, smoothed, start_frame)
    np.save(output_dir / "motionagformer_h36m17.npy", raw)
    np.savez_compressed(artifact, frames=frames, body18_2d=smoothed,
                        confidence=confidence, h36m17_3d=pose,
                        pixels_per_unit=np.float32(pixels_per_unit),
                        bone_lengths=bone_lengths)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return artifact


def lift_openpose_json(*args, **kwargs):
    """Backward-compatible alias for :func:`lift_openpose_sequence`."""
    return lift_openpose_sequence(*args, **kwargs)
