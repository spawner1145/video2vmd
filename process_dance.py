#!/usr/bin/env python
"""Run the SDPose, UniDepth, MotionAGFormer, and VMD conversion pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
BODY18_JOINT_NAMES = (
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
    "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee",
    "LAnkle", "REye", "LEye", "REar", "LEar",
)
BODY18_EDGES = (
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
    (0, 14), (14, 16), (0, 15), (15, 17),
)
H36M_EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_video", nargs="?", type=Path,
        default=PROJECT_ROOT / "data" / "dance.mp4")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output" / "dance_pipeline")
    parser.add_argument("--openpose-model-dir", type=Path,
                        default=PROJECT_ROOT / "OpenPose_PyTorch" / "model")
    parser.add_argument("--sdpose-model-dir", type=Path,
                        default=PROJECT_ROOT / "SDPose_body")
    parser.add_argument("--unidepth-model-path", type=Path,
                        default=PROJECT_ROOT / "unidepth-v2-vitl14")
    parser.add_argument("--motionagformer-checkpoint", type=Path,
                        default=PROJECT_ROOT / "motionagformer_b_h36m.pth")
    parser.add_argument(
        "--stage", choices=("all", "pose", "depth", "lift", "motion", "export"),
        default="all")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--depth-interval", "--depth-stride", dest="depth_interval",
        type=int, default=3, help="Run depth inference every N frames")
    parser.add_argument(
        "--max-people", "--number-people-max", dest="max_people",
        type=int, default=1, help="Maximum number of people to detect and track")
    parser.add_argument(
        "--start-frame", "--frame-first", dest="start_frame",
        type=int, default=0, help="Zero-based source frame at which processing starts")
    parser.add_argument(
        "--person-index", type=int, default=0,
        help="Tracked person index used for 3D and VMD export")
    parser.add_argument("--depth-backend", choices=("unidepth", "none"), default="unidepth")
    parser.add_argument("--unidepth-resolution-level", "--unidepth-level", dest="unidepth_resolution_level", type=int, default=0)
    parser.add_argument("--vmd-center-z-scale", type=float, default=5.0)
    parser.add_argument("--vmd-smoothing-passes", "--vmd-smooth", dest="vmd_smoothing_passes", type=int, default=1)
    parser.add_argument("--vmd-position-threshold", "--vmd-reduce-pos", dest="vmd_position_threshold", type=float, default=0.0)
    parser.add_argument("--vmd-rotation-threshold", "--vmd-reduce-rot", dest="vmd_rotation_threshold", type=float, default=0.0)
    parser.add_argument(
        "--disable-foot-ik", "--vmd-fk", dest="disable_foot_ik",
        action="store_true", help="Disable foot IK in the original VMD converter")
    parser.add_argument("--force", action="store_true", help="Recompute cached stages")
    parser.add_argument("--pose-source", choices=("adaptive", "openpose", "sdpose-limbs"), default="sdpose-limbs",
                        help="Adaptively fuse OpenPose/SDPose, or select one limb source")
    parser.add_argument("--lifting-backend", choices=("motionagformer", "pose-depth"), default="motionagformer",
                        help="Use MotionAGFormer lifting or direct 2D pose plus UniDepth reconstruction")
    parser.add_argument("--flip-depth-joints", default="",
                        help="Comma-separated BODY18 joint indexes whose Z direction is reflected, e.g. 8,11")
    parser.add_argument("--head-facing-camera", action="store_true",
                        help="Cancel upper-body rotation on VMD neck/head so the face stays camera-forward")
    return parser.parse_args()


def read_video_metadata(video_path, max_frames=None, start_frame=0):
    capture = cv2.VideoCapture(str(video_path))
    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame)
    capture.release()
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    return frames_per_second, width, height, frame_count


def estimate_2d_pose(arguments, frame_count):
    from pose_detection_openpose import extract_openpose_keypoints

    if arguments.pose_source == "adaptive":
        return estimate_adaptive_pose(arguments, frame_count)
    if arguments.pose_source == "sdpose-limbs":
        return estimate_sdpose_full(arguments, frame_count)
    pose_cache = arguments.output_dir / "openpose_body18.npz"
    if pose_cache.exists() and not arguments.force:
        return
    preview_name = f"{arguments.input_video.stem}_pose_preview.avi"
    extract_openpose_keypoints(
        arguments.input_video,
        arguments.output_dir / "openpose_json",
        arguments.output_dir / preview_name,
        model_dir=arguments.openpose_model_dir,
        max_people=arguments.max_people,
        start_frame=arguments.start_frame,
        max_frames=frame_count,
        cache_path=pose_cache,
        force=arguments.force,
    )


def estimate_adaptive_pose(arguments, frame_count):
    """Run both detectors and confidence-blend only the BODY18 limb joints."""
    from pose_detection_openpose import extract_openpose_keypoints
    native_cache = arguments.output_dir / "openpose_body18.npz"
    native_json = arguments.output_dir / "openpose_native_json"
    native_json.mkdir(parents=True, exist_ok=True)
    if not native_cache.exists() or arguments.force:
        extract_openpose_keypoints(
            arguments.input_video, native_json,
            arguments.output_dir / "openpose_native_preview.avi",
            model_dir=arguments.openpose_model_dir,
            max_people=arguments.max_people, start_frame=arguments.start_frame,
            max_frames=frame_count, cache_path=native_cache, force=arguments.force)
    sd_model = arguments.sdpose_model_dir
    sd_cache = arguments.output_dir / "sdpose_coco17.npz"
    sd_json = arguments.output_dir / "sdpose_json"
    sd_json.mkdir(parents=True, exist_ok=True)
    if not (sd_model / "decoder" / "decoder.safetensors").exists():
        # Explicit fallback when SDPose is unavailable.
        shutil.copytree(native_json, arguments.output_dir / "openpose_json", dirs_exist_ok=True)
        shutil.copy2(native_cache, arguments.output_dir / "openpose_body18.npz")
        return
    from pose_detection_openpose.sdpose_backend import extract_openpose_keypoints as extract_sdpose
    if not sd_cache.exists() or arguments.force:
        extract_sdpose(
            arguments.input_video, sd_json,
            arguments.output_dir / "sdpose_preview.avi", model_dir=sd_model,
            max_people=arguments.max_people, start_frame=arguments.start_frame,
            max_frames=frame_count, cache_path=sd_cache, force=arguments.force)
    native = np.load(native_cache); sd = np.load(sd_cache)
    native_points, native_conf = native["body18"], native["confidence"]
    sd_points, sd_conf = sd["keypoints"], sd["confidence"]
    # BODY18 target -> SDPose COCO17 source (right/left semantics included).
    mapping = {0: 0, 14: 1, 15: 2, 16: 3, 17: 4,
               2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9,
               8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15}
    body, confidence = native_points.copy(), native_conf.copy()
    # SDPose contributes up to 35% when confident, and takes over when
    # OpenPose is missing or clearly unreliable for that limb.
    for target, source in mapping.items():
        sd_xy = sd_points[:len(body), source]
        sc, oc = sd_conf[:len(body), source], confidence[:, target]
        weight = np.clip((sc - oc + .15) / .6, 0, .35)
        missing = oc < .25
        weight[missing] = np.maximum(weight[missing], .8)
        body[:, target] = body[:, target] * (1 - weight[:, None]) + sd_xy * weight[:, None]
        confidence[:, target] = np.maximum(oc, sc * weight)
    # COCO17 has no explicit neck; derive it from the two SDPose shoulders.
    sd_neck = (sd_points[:len(body), 5] + sd_points[:len(body), 6]) * .5
    sd_neck_conf = np.minimum(sd_conf[:len(body), 5], sd_conf[:len(body), 6])
    weight = np.clip((sd_neck_conf - confidence[:, 1] + .15) / .6, 0, .35)
    body[:, 1] = body[:, 1] * (1 - weight[:, None]) + sd_neck * weight[:, None]
    confidence[:, 1] = np.maximum(confidence[:, 1], sd_neck_conf * weight)
    face = body[:, [0, 14, 15, 16, 17]]
    body[:, [0, 14, 15, 16, 17]] = median_filter(face, size=(11, 1, 1), mode="nearest")
    merged_cache = arguments.output_dir / "openpose_body18.npz"
    np.savez_compressed(merged_cache, body18=body, confidence=confidence,
                        start_frame=native.get("start_frame", arguments.start_frame))
    # Copy native JSON then replace its BODY18 payload with blended points.
    target_json = arguments.output_dir / "openpose_json"
    target_json.mkdir(parents=True, exist_ok=True)
    files = sorted(native_json.glob("*.json"))
    for index, source_file in enumerate(files[:len(body)]):
        payload = {"version": 1.3, "people": [{"person_id": [-1],
            "pose_keypoints_2d": np.c_[body[index], confidence[index]].reshape(-1).tolist(),
            "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []}]}
        (target_json / source_file.name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def estimate_hybrid_sdpose_limbs(arguments, frame_count, pose_cache):
    """Merge cached SDPose COCO17 limbs with native OpenPose face/neck points."""
    source_json = arguments.output_dir / "openpose_json"
    native_cache = arguments.output_dir / "openpose_body18.npz"
    sd_cache = arguments.output_dir / "sdpose_coco17.npz"
    if not native_cache.exists():
        raise FileNotFoundError("Hybrid mode requires output/dance_pipeline/openpose_body18.npz")
    if not sd_cache.exists():
        raise FileNotFoundError("Hybrid mode requires output/dance_pipeline/sdpose_coco17.npz")
    if pose_cache.exists() and not arguments.force:
        return
    native = np.load(native_cache)
    sd = np.load(sd_cache)
    native_points = native["body18"]
    native_conf = native["confidence"]
    if "keypoints" in sd:
        sd_points, sd_conf = sd["keypoints"], sd["confidence"]
    else:
        sd_points, sd_conf = sd["points"], sd["scores"]
    count = min(frame_count, len(native_points), len(sd_points))
    # COCO17 -> BODY18 indices from pose_detection_openpose.sdpose_backend.
    mapping = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9,
               8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
               14: 2, 15: 1, 16: 4, 17: 3}
    body = native_points[:count].copy(); confidence = native_conf[:count].copy()
    for target, source in mapping.items():
        if target in (0, 1, 14, 15, 16, 17):
            continue
        body[:, target] = sd_points[:count, source]
        confidence[:, target] = sd_conf[:count, source]
    json_dir = source_json
    json_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(count):
        values = np.c_[body[frame], confidence[frame]].reshape(-1).tolist()
        payload = {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": values,
            "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []}]}
        name = f"{arguments.input_video.stem}_{frame:012d}_keypoints.json"
        (json_dir / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(pose_cache, body18=body, confidence=confidence, start_frame=arguments.start_frame)


def estimate_depth_and_track_people(arguments, frame_count):
    if arguments.depth_backend == "none":
        return
    from depth_tracking_vmd import estimate_depth_and_track_people as run_depth_tracking

    run_depth_tracking(
        arguments.input_video,
        arguments.output_dir / "openpose_json",
        arguments.output_dir,
        model_path=arguments.unidepth_model_path,
        depth_interval=arguments.depth_interval,
        max_people=arguments.max_people,
        resolution_level=arguments.unidepth_resolution_level,
        end_frame_no=frame_count,
        force=arguments.force,
    )


def lift_pose_to_3d(arguments, video_width, video_height):
    if arguments.lifting_backend == "pose-depth":
        return
    from pose_baseline_vmd import lift_openpose_sequence

    run_name = arguments.output_dir.name.removesuffix("_depth")
    tracked_json_dir = (
        arguments.output_dir
        / f"openpose_json_{run_name}_idx{arguments.person_index + 1:02d}"
        / "json"
    )
    input_json_dir = (arguments.output_dir / "openpose_json" if arguments.force
                      else tracked_json_dir if tracked_json_dir.exists()
                      else arguments.output_dir / "openpose_json")
    lift_openpose_sequence(
        input_json_dir,
        arguments.output_dir,
        video_width=video_width,
        video_height=video_height,
        checkpoint=arguments.motionagformer_checkpoint,
        force=arguments.force,
    )


def reconstruct_pose_depth_3d(body18_2d, depth, width, height):
    """Direct 2D+depth reconstruction in the millimetre contract used by VMD."""
    if depth is None or len(depth) != len(body18_2d):
        raise ValueError("pose-depth backend requires per-frame UniDepth joint depths")
    pixels = body18_2d.astype(np.float32)
    root_2d = (pixels[:, 8] + pixels[:, 11]) * .5
    root_depth = (depth[:, 8] + depth[:, 11]) * .5
    relative_depth = depth - root_depth[:, None]
    # Metric depth has an arbitrary global scale for this camera; retain its
    # relative ordering and calibrate it to the observed 2D body scale.
    shoulder_px = np.linalg.norm(pixels[:, 2] - pixels[:, 5], axis=1)
    hip_px = np.linalg.norm(pixels[:, 8] - pixels[:, 11], axis=1)
    image_scale = np.median(np.r_[shoulder_px[shoulder_px > 2], hip_px[hip_px > 2]])
    xy_scale = 360.0 / max(float(image_scale), 1.0)
    z_scale = xy_scale * max(float(np.median(np.abs(relative_depth))), 1e-4) ** -1 * 45.0
    pose = np.zeros((len(pixels), 18, 3), np.float32)
    pose[:, :, 0] = (pixels[:, :, 0] - root_2d[:, None, 0]) * xy_scale
    pose[:, :, 1] = -(pixels[:, :, 1] - root_2d[:, None, 1]) * xy_scale
    pose[:, :, 2] = relative_depth * z_scale
    pose[:, :, 2] = median_filter(pose[:, :, 2], size=(5, 1), mode="nearest")
    return pose
def convert_h36m17_to_body18(h36m_pose, body18_2d=None, pixels_per_unit=None):
    body18_pose = np.zeros((len(h36m_pose), 18, 3), np.float32)
    body18_pose[:, [8, 9, 10]] = h36m_pose[:, [4, 5, 6]]
    body18_pose[:, [11, 12, 13]] = h36m_pose[:, [1, 2, 3]]
    body18_pose[:, 1] = h36m_pose[:, 8]
    body18_pose[:, 0] = h36m_pose[:, 10]
    body18_pose[:, [2, 3, 4]] = h36m_pose[:, [11, 12, 13]]
    body18_pose[:, [5, 6, 7]] = h36m_pose[:, [14, 15, 16]]
    shoulder_axis = h36m_pose[:, 11] - h36m_pose[:, 14]
    shoulder_width = np.linalg.norm(shoulder_axis, axis=1, keepdims=True)
    lateral = shoulder_axis / np.maximum(shoulder_width, 1e-6)
    nose = body18_pose[:, 0]
    body18_pose[:, 14] = nose + lateral * shoulder_width * .055
    body18_pose[:, 15] = nose - lateral * shoulder_width * .055
    body18_pose[:, 16] = nose + lateral * shoulder_width * .105
    body18_pose[:, 17] = nose - lateral * shoulder_width * .105
    if body18_2d is not None and pixels_per_unit:
        # Restore the real OpenPose face Y in converter-facing X/up coordinates.
        # UniDepth fusion supplies each face point's depth afterwards.
        face_offset = (body18_2d[:, 14:18] - body18_2d[:, 0:1]) / pixels_per_unit
        body18_pose[:, 14:18, 0] = nose[:, None, 0] + face_offset[:, :, 0]
        body18_pose[:, 14:18, 1] = nose[:, None, 1] - face_offset[:, :, 1]
    return body18_pose


def write_vmd_depth_files(output_dir, joint_depth, joint_confidence, start_frame):
    frame_indexes = np.arange(len(joint_depth), dtype=np.float32)[:, None]
    hip_depth = (joint_depth[:, 8] + joint_depth[:, 11]) / 2
    relative_joint_depth = joint_depth[:, :18] - np.median(hip_depth)
    segment_depth = np.stack([
        (relative_joint_depth[:, start] + relative_joint_depth[:, end]) / 2
        for start, end in BODY18_EDGES
    ], axis=1)
    segment_confidence = np.stack([
        (joint_confidence[:, start] + joint_confidence[:, end]) / 2
        for start, end in BODY18_EDGES
    ], axis=1)
    np.savetxt(
        output_dir / "depth.txt",
        np.c_[frame_indexes, relative_joint_depth, segment_depth],
        fmt="%.8f", delimiter=",")
    confidence_rows = np.c_[frame_indexes, joint_confidence[:, :18], segment_confidence]
    np.savetxt(output_dir / "depth_conf.txt", confidence_rows, fmt="%.8f", delimiter=",")
    np.savetxt(output_dir / "conf.txt", confidence_rows, fmt="%.8f", delimiter=",")
    (output_dir / "start_frame.txt").write_text(str(start_frame), encoding="ascii")


def write_pose_depth_vmd_positions(output_dir, body18_3d, body18_2d, start_frame):
    original_indices = (0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27)
    pose = np.zeros((len(body18_3d), 17, 3), np.float32)
    pose[:, 0] = (body18_3d[:, 8] + body18_3d[:, 11]) * .5
    pose[:, 1:4] = body18_3d[:, [8, 9, 10]]
    pose[:, 4:7] = body18_3d[:, [11, 12, 13]]
    pose[:, 8] = body18_3d[:, 1]
    pose[:, 7] = (pose[:, 0] + pose[:, 8]) * .5
    pose[:, 9] = body18_3d[:, 0]
    pose[:, 10] = (body18_3d[:, 16] + body18_3d[:, 17]) * .5
    pose[:, 11:14] = body18_3d[:, [2, 3, 4]]
    pose[:, 14:17] = body18_3d[:, [5, 6, 7]]
    with (output_dir / "pos.txt").open("w", encoding="ascii") as handle:
        for frame_pose in pose:
            handle.write(", ".join(
                f"{index} {joint[0]:.8f} {joint[2]:.8f} {joint[1]:.8f}"
                for index, joint in zip(original_indices, frame_pose)) + ", \n")
    np.savetxt(output_dir / "smoothed.txt", body18_2d.reshape(len(body18_2d), 36), fmt="%.8f")
    (output_dir / "start_frame.txt").write_text(str(int(start_frame)), encoding="ascii")


def generate_vmd(arguments, output_dir=None, motion_name=None):
    converter_root = PROJECT_ROOT / "VMD_3d_pose_baseline_multi"
    output_dir = Path(output_dir or arguments.output_dir).resolve()
    previous_working_directory = Path.cwd()
    sys.path.insert(0, str(converter_root))
    try:
        os.chdir(converter_root)
        from applications import pos2vmd_multi, pos2vmd_utils
        from applications import pos2vmd_frame
        pos2vmd_frame.FORCE_HEAD_FACING_CAMERA = arguments.head_facing_camera

        for bone_frames in pos2vmd_multi.bone_frame_dic.values():
            bone_frames.clear()
        positions = pos2vmd_utils.read_positions_multi(str(output_dir / "pos.txt"))
        motion_name = motion_name or f"{arguments.input_video.stem}_motion"
        output_template = str(output_dir / f"{motion_name}_[type].vmd")
        pos2vmd_multi.position_list_to_vmd_multi(
            positions,
            output_template,
            str(output_dir / "smoothed.txt"),
            str(converter_root / "born" / "animasa_miku_born.csv"),
            str(output_dir / "depth.txt"),
            str(output_dir / "depth_conf.txt"),
            str(output_dir / "start_frame.txt"),
            30,
            arguments.vmd_center_z_scale,
            1,
            arguments.vmd_smoothing_passes,
            arguments.vmd_position_threshold,
            arguments.vmd_rotation_threshold,
            not arguments.disable_foot_ik,
            0,
            str(output_dir),
            "pipeline",
        )
        output_type = "reduce" if arguments.vmd_position_threshold or arguments.vmd_rotation_threshold else "full"
        return Path(output_template.replace("[type]", output_type))
    finally:
        os.chdir(previous_working_directory)


def write_fused_vmd_inputs(source_dir, fused_body18, baseline_body18):
    """Copy original converter inputs and apply fused depth to matching BODY18 joints."""
    fused_dir = source_dir / "fused_vmd_inputs"
    fused_dir.mkdir(parents=True, exist_ok=True)
    for name in ("smoothed.txt", "depth.txt", "depth_conf.txt", "start_frame.txt"):
        shutil.copy2(source_dir / name, fused_dir / name)

    depth_delta = fused_body18[:, :, 2] - baseline_body18[:, :, 2]
    direct_mapping = {
        1: 8, 2: 9, 3: 10, 6: 11, 7: 12, 8: 13,
        13: 1, 14: 0,
        17: 2, 18: 3, 19: 4, 25: 5, 26: 6, 27: 7,
    }
    derived_mapping = {
        0: lambda delta: (delta[8] + delta[11]) * .5,
        12: lambda delta: (delta[1] + delta[8] + delta[11]) / 3,
        15: lambda delta: (delta[16] + delta[17]) * .5,
    }
    output_lines = []
    for frame_index, line in enumerate((source_dir / "pos.txt").read_text(encoding="ascii").splitlines()):
        entries = []
        for entry in line.split(","):
            fields = entry.strip().split()
            if len(fields) != 4:
                continue
            joint_index = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if joint_index in direct_mapping:
                values[1] += float(depth_delta[frame_index, direct_mapping[joint_index]])
            elif joint_index in derived_mapping:
                values[1] += float(derived_mapping[joint_index](depth_delta[frame_index]))
            entries.append(f"{joint_index} {values[0]:.8f} {values[1]:.8f} {values[2]:.8f}")
        output_lines.append(", ".join(entries) + ", ")
    (fused_dir / "pos.txt").write_text("\n".join(output_lines) + "\n", encoding="ascii")
    return fused_dir


def estimate_sdpose_full(arguments, frame_count):
    """Use SDPose COCO17 alone, converting it to the BODY18 contract."""
    from pose_detection_openpose.sdpose_backend import coco17_to_body18
    model_dir = arguments.sdpose_model_dir
    sd_cache = arguments.output_dir / "sdpose_coco17.npz"
    sd_json = arguments.output_dir / "sdpose_json"
    body_cache = arguments.output_dir / "openpose_body18.npz"
    if not (model_dir / "decoder" / "decoder.safetensors").exists():
        # Pure SDPose is the default, but remain usable on a checkout without
        # its weights by falling back to the native OpenPose backend.
        arguments.pose_source = "openpose"
        return estimate_2d_pose(arguments, frame_count)
    if not sd_cache.exists() or arguments.force:
        from pose_detection_openpose.sdpose_backend import extract_openpose_keypoints
        extract_openpose_keypoints(
            arguments.input_video, sd_json, arguments.output_dir / "sdpose_preview.avi",
            model_dir=model_dir, max_people=arguments.max_people,
            start_frame=arguments.start_frame, max_frames=frame_count,
            cache_path=sd_cache, force=arguments.force)
    sd = np.load(sd_cache)
    coco_points, coco_conf = sd["keypoints"], sd["confidence"]
    count = min(frame_count, len(coco_points))
    body = np.zeros((count, 18, 2), np.float32)
    confidence = np.zeros((count, 18), np.float32)
    for index in range(count):
        body[index], confidence[index] = coco17_to_body18(
            coco_points[index], coco_conf[index])
    body[:, [0, 14, 15, 16, 17]] = median_filter(
        body[:, [0, 14, 15, 16, 17]], size=(11, 1, 1), mode="nearest")
    np.savez_compressed(body_cache, body18=body, confidence=confidence,
                        start_frame=sd.get("start_frame", arguments.start_frame))
    json_dir = arguments.output_dir / "openpose_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        payload = {"version": 1.3, "people": [{"person_id": [-1],
            "pose_keypoints_2d": np.c_[body[index], confidence[index]].reshape(-1).tolist(),
            "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []}]}
        name = f"{arguments.input_video.stem}_{index:012d}_keypoints.json"
        (json_dir / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def interpolate_metric_depth(arguments, frame_count):
    cache_path = arguments.output_dir / "unidepth_metric.npz"
    if not cache_path.exists():
        return None, None
    cache = np.load(cache_path)
    target_frames = np.arange(frame_count) + arguments.start_frame
    joint_depth = np.stack([
        np.interp(target_frames, cache["frames"], cache["depth_m"][:, joint])
        for joint in range(18)
    ], axis=1)
    joint_confidence = np.stack([
        np.interp(target_frames, cache["frames"], cache["confidence"][:, joint])
        for joint in range(18)
    ], axis=1)
    confidence_scale = max(float(np.percentile(joint_confidence, 95)), 1e-6)
    joint_confidence = np.clip(joint_confidence / confidence_scale, 0, 1)
    joint_depth = median_filter(joint_depth, size=(7, 1), mode="nearest")
    return joint_depth.astype(np.float32), joint_confidence.astype(np.float32)


def fuse_openpose_motionag_unidepth(body18_pose, body18_2d, depth, depth_confidence):
    """Fuse 2D observations, MotionAGFormer 3D, and UniDepth Z by constrained LS."""
    if depth is None or not len(depth):
        return body18_pose.copy()
    fused = body18_pose.astype(np.float32).copy()
    hip_depth = (depth[:, 8] + depth[:, 11]) / 2
    depth_relative = depth - hip_depth[:, None]
    depth_relative = median_filter(depth_relative, size=(7, 1), mode="nearest")
    pose_z = fused[:, :, 2]
    pose_center = np.median(pose_z, axis=1, keepdims=True)
    pose_scale = np.median(np.abs(pose_z - pose_center), axis=1, keepdims=True) + 1e-5
    depth_center = np.median(depth_relative, axis=1, keepdims=True)
    depth_scale = np.median(np.abs(depth_relative - depth_center), axis=1, keepdims=True) + 1e-5
    depth_target = (depth_relative - depth_center) * pose_scale / depth_scale + pose_center
    # UniDepth is useful as a global depth cue, but its per-joint metric noise
    # can invert a real torso lean. Keep MotionAGFormer dominant for hips/body
    # and apply stronger temporal filtering to the face direction.
    depth_target = median_filter(depth_target, size=(11, 1), mode="nearest")
    depth_target[:, [0, 14, 15, 16, 17]] = median_filter(
        depth_target[:, [0, 14, 15, 16, 17]], size=(21, 1), mode="nearest")
    edges = BODY18_EDGES
    reference_lengths = np.median(np.linalg.norm(
        fused[:, [end for _, end in edges]] - fused[:, [start for start, _ in edges]], axis=2), axis=0)
    for frame in range(len(fused)):
        confidence = np.clip(depth_confidence[frame], 0, 1)
        x, y = fused[frame, :, 0], fused[frame, :, 1]
        initial = fused[frame, :, 2].copy()

        observation_weight = np.full(18, .32, np.float32)
        observation_weight[[8, 11, 1]] = .22
        observation_weight[[0, 14, 15, 16, 17]] = .20

        def residual(z):
            values = [np.sqrt(.35) * (z - initial)]
            values.append(np.sqrt(observation_weight) * confidence * (z - depth_target[frame]))
            bone_errors = []
            for edge_index, (start, end) in enumerate(edges):
                length = np.sqrt((x[end] - x[start]) ** 2 + (y[end] - y[start]) ** 2 + (z[end] - z[start]) ** 2)
                bone_errors.append((length - reference_lengths[edge_index]) * .25)
            values.append(np.asarray(bone_errors, np.float32))
            return np.concatenate(values)

        fused[frame, :, 2] = least_squares(residual, initial, max_nfev=20).x
    fused[:, :, 2] = median_filter(fused[:, :, 2], size=(5, 1), mode="nearest")
    return fused


def suppress_short_pose_spikes(pose, max_run=2):
    """Replace isolated 1-2 frame coordinate spikes without smoothing real motion."""
    filtered = pose.astype(np.float32).copy()
    local_median = median_filter(filtered, size=(5, 1, 1), mode="nearest")
    residual = np.abs(filtered - local_median)
    local_mad = median_filter(residual, size=(11, 1, 1), mode="nearest")
    body_scale = np.median(np.linalg.norm(filtered[:, 1] - (filtered[:, 8] + filtered[:, 11]) / 2, axis=1))
    threshold = np.maximum(6.0 * local_mad, max(body_scale * .018, 2.0))
    candidates = residual > threshold
    # Face landmarks are small and directly control head direction, so reject
    # smaller isolated errors there while retaining the same run-length rule.
    candidates[:, 14:18] |= residual[:, 14:18] > np.maximum(
        4.0 * local_mad[:, 14:18], max(body_scale * .009, 1.0))
    candidates[:2] = False
    candidates[-2:] = False

    corrected = np.zeros_like(candidates)
    for joint in range(filtered.shape[1]):
        for dimension in range(3):
            indexes = np.flatnonzero(candidates[:, joint, dimension])
            if not len(indexes):
                continue
            for run in np.split(indexes, np.where(np.diff(indexes) != 1)[0] + 1):
                if len(run) <= max_run:
                    corrected[run, joint, dimension] = True
            bad = corrected[:, joint, dimension]
            good = ~bad
            if bad.any() and good.sum() >= 2:
                filtered[bad, joint, dimension] = np.interp(
                    np.flatnonzero(bad), np.flatnonzero(good), filtered[good, joint, dimension])
    return filtered, int(corrected.sum())


def apply_depth_joint_flips(pose, specification):
    indexes = [] if not specification.strip() else [int(value.strip()) for value in specification.split(",")]
    invalid = [index for index in indexes if not 0 <= index < 18]
    if invalid:
        raise ValueError(f"BODY18 joint indexes must be in 0..17, got {invalid}")
    if not indexes:
        return pose, indexes
    result = pose.copy()
    torso_center = np.median(result[:, [1, 2, 5, 8, 11], 2], axis=1)
    result[:, indexes, 2] = 2 * torso_center[:, None] - result[:, indexes, 2]
    return result, indexes


def project_pose_for_preview(pose, view, angle=0.0, elevation=0.0):
    centered_pose = pose - (pose[8] + pose[11]) / 2
    if view == "front":
        return centered_pose[:, [0, 1]] * np.array([1, -1])
    if view == "side":
        return centered_pose[:, [2, 1]] * np.array([1, -1])
    horizontal = centered_pose[:, 0] * np.cos(angle) + centered_pose[:, 2] * np.sin(angle)
    view_depth = -centered_pose[:, 0] * np.sin(angle) + centered_pose[:, 2] * np.cos(angle)
    vertical = -centered_pose[:, 1] * np.cos(elevation) + view_depth * np.sin(elevation)
    return np.c_[horizontal, vertical]


def draw_3d_preview_panel(canvas, pose, bounds, label, view, angle=0.0, elevation=0.0):
    left, top, width, height = bounds
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (55, 60, 68), 1)
    cv2.putText(canvas, label, (left + 15, top + 30), 0, .7, (230, 230, 230), 2)
    projected = project_pose_for_preview(pose, view, angle, elevation)
    scale = min(width, height) * .72 / max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]), 1e-4)
    projected = projected * scale + [left + width / 2, top + height * .55]
    for start, end in BODY18_EDGES:
        cv2.line(canvas, tuple(projected[start].astype(int)), tuple(projected[end].astype(int)), (80, 210, 245), 4, cv2.LINE_AA)
    for point in projected:
        cv2.circle(canvas, tuple(point.astype(int)), 4, (245, 245, 245), -1)


def draw_numbered_3d_pose(pose, width=960, height=720):
    canvas = np.full((height, width, 3), 255, np.uint8)
    # MotionAGFormer stores X/up/depth; the original viz.py plots X/depth/up
    # and calls those axes X/Y/Z respectively.
    pose = pose[:, [0, 2, 1]]
    centered = pose - pose[0]
    yaw, elevation = np.deg2rad(45), np.deg2rad(22)

    def project(points):
        horizontal = points[:, 0] * np.cos(yaw) + points[:, 1] * np.sin(yaw)
        view_depth = -points[:, 0] * np.sin(yaw) + points[:, 1] * np.cos(yaw)
        vertical = -points[:, 2] * np.cos(elevation) + view_depth * np.sin(elevation)
        return np.c_[horizontal, vertical]

    body_projected = project(centered)
    span = max(float(np.ptp(centered[:, 0])), float(np.ptp(centered[:, 1])), .5) * .72
    ground_y = float(np.min(centered[:, 2]))
    ground = np.asarray([
        [-span, -span, ground_y], [span, -span, ground_y],
        [span, span, ground_y], [-span, span, ground_y],
    ], np.float32)
    ground_projected = project(ground)
    combined = np.vstack([body_projected, ground_projected])
    scale = min(width * .68 / max(np.ptp(combined[:, 0]), 1e-4),
                height * .72 / max(np.ptp(combined[:, 1]), 1e-4))
    offset = np.asarray([width * .5, height * .5]) - combined.mean(axis=0) * scale
    body_projected = body_projected * scale + offset
    ground_projected = ground_projected * scale + offset

    cv2.fillConvexPoly(canvas, ground_projected.astype(np.int32), (242, 242, 242), cv2.LINE_AA)
    cv2.polylines(canvas, [ground_projected.astype(np.int32)], True, (225, 225, 225), 2, cv2.LINE_AA)

    right_joints = {2, 3, 4, 8, 9, 10, 14, 16}
    left_joints = {5, 6, 7, 11, 12, 13, 15, 17}
    for start, end in BODY18_EDGES:
        if start in right_joints and end in right_joints:
            color = (55, 55, 230)
        elif start in left_joints and end in left_joints:
            color = (220, 70, 35)
        else:
            color = (45, 45, 45)
        cv2.line(canvas, tuple(body_projected[start].astype(int)),
                 tuple(body_projected[end].astype(int)), color, 4, cv2.LINE_AA)

    label_offsets = {
        0: (8, -9), 1: (8, 4), 2: (-24, -7), 3: (-24, 3),
        4: (-24, 14), 5: (8, -7), 6: (8, 3), 7: (8, 14),
        8: (-24, -7), 9: (-24, 3), 10: (-27, 14),
        11: (8, -7), 12: (8, 3), 13: (8, 14),
        14: (-28, -13), 15: (10, -13), 16: (-32, 8), 17: (10, 8),
    }
    for joint, point in enumerate(body_projected):
        location = tuple(point.astype(int))
        cv2.circle(canvas, location, 5, (30, 30, 30), -1, cv2.LINE_AA)
        dx, dy = label_offsets.get(joint, (7, -7))
        cv2.putText(canvas, str(joint), (location[0] + dx, location[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (15, 15, 15), 1, cv2.LINE_AA)

    cv2.putText(canvas, "X", (90, height - 55), cv2.FONT_HERSHEY_SIMPLEX, .8, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Y", (width - 115, height - 55), cv2.FONT_HERSHEY_SIMPLEX, .8, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Z", (width - 85, 85), cv2.FONT_HERSHEY_SIMPLEX, .8, (25, 25, 25), 2, cv2.LINE_AA)
    return canvas


def add_source_audio(silent_video, source_video, output_video):
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(silent_video), "-i", str(source_video),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
        "-c:a", "aac", "-shortest", str(output_video),
    ], check=True)
    silent_video.unlink()


def export_results(arguments, frames_per_second, video_width, video_height, frame_count):
    if arguments.lifting_backend == "pose-depth":
        pose_cache = np.load(arguments.output_dir / "openpose_body18.npz")
        body18_2d = pose_cache["body18"]
        pose_confidence = pose_cache["confidence"]
        h36m17_3d = np.zeros((len(body18_2d), 17, 3), np.float32)
        pixels_per_unit = 1.0
        bone_lengths = np.zeros(len(H36M_EDGES), np.float32)
    else:
        lifting_result = np.load(arguments.output_dir / "motionagformer_pose.npz")
        body18_2d = lifting_result["body18_2d"]
        pose_confidence = lifting_result["confidence"]
        h36m17_3d = lifting_result["h36m17_3d"]
        pixels_per_unit = float(lifting_result["pixels_per_unit"])
        bone_lengths = lifting_result["bone_lengths"]

    metric_depth, metric_confidence = interpolate_metric_depth(arguments, frame_count)
    if metric_depth is None:
        joint_depth = np.zeros((frame_count, 18), np.float32)
        depth_confidence = np.zeros((frame_count, 18), np.float32)
        depth_description = "disabled"
        root_depth_offset = np.zeros(frame_count, np.float32)
    else:
        joint_depth = metric_depth
        depth_confidence = metric_confidence * pose_confidence[:, :18]
        root_depth = (metric_depth[:, 8] + metric_depth[:, 11]) / 2
        if len(root_depth) >= 5:
            window_length = min(31, len(root_depth) if len(root_depth) % 2 else len(root_depth) - 1)
            root_depth = savgol_filter(root_depth, window_length, min(2, window_length - 1), mode="interp")
        root_depth_offset = np.clip(root_depth - np.median(root_depth), -2, 2)
        depth_description = "UniDepth V2 metric depth for VMD center Z"

    if arguments.lifting_backend == "pose-depth":
        body18_3d = reconstruct_pose_depth_3d(
            body18_2d, joint_depth, video_width, video_height)
        pixels_per_unit = float(1.0 / max(np.median(np.linalg.norm(
            body18_2d[:, 2] - body18_2d[:, 5], axis=1)), 1e-6))
        body18_3d, spike_count = suppress_short_pose_spikes(body18_3d)
    else:
        body18_3d = convert_h36m17_to_body18(h36m17_3d, body18_2d, pixels_per_unit)
        body18_3d[:, :, 2] += root_depth_offset[:, None]
        spike_count = 0

    if arguments.lifting_backend == "pose-depth":
        fused_body18_3d = body18_3d
    else:
        fused_body18_3d = fuse_openpose_motionag_unidepth(
            body18_3d, body18_2d, joint_depth, depth_confidence)
        fused_body18_3d, spike_count = suppress_short_pose_spikes(fused_body18_3d)
    fused_body18_3d, flipped_depth_joints = apply_depth_joint_flips(
        fused_body18_3d, arguments.flip_depth_joints)

    if arguments.lifting_backend == "pose-depth":
        start_frame = int(pose_cache["start_frame"])
        write_pose_depth_vmd_positions(arguments.output_dir, body18_3d, body18_2d, start_frame)
    else:
        start_frame = int(lifting_result["frames"][0])
    write_vmd_depth_files(arguments.output_dir, joint_depth, depth_confidence, start_frame)
    vmd_path = generate_vmd(arguments)
    fused_vmd_dir = write_fused_vmd_inputs(arguments.output_dir, fused_body18_3d, body18_3d)
    fused_vmd_path = generate_vmd(
        arguments, output_dir=fused_vmd_dir,
        motion_name=f"{arguments.input_video.stem}_motion_fused")
    np.savez_compressed(
        arguments.output_dir / "pose_data.npz",
        body18_2d=body18_2d,
        confidence=pose_confidence,
        h36m17_3d=h36m17_3d,
        body18_3d=body18_3d,
        fused_body18_3d=fused_body18_3d,
        depth=joint_depth,
        depth_confidence=depth_confidence,
        pixels_per_unit=pixels_per_unit,
        bone_lengths=bone_lengths,
        fps=frames_per_second,
        joint_names=np.asarray(BODY18_JOINT_NAMES),
    )

    capture = cv2.VideoCapture(str(arguments.input_video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, arguments.start_frame)
    silent_2d_path = arguments.output_dir / "pose_2d_preview_silent.mp4"
    writer = cv2.VideoWriter(str(silent_2d_path), cv2.VideoWriter_fourcc(*"mp4v"), frames_per_second, (video_width, video_height))
    for frame_index in tqdm(range(frame_count), desc="Rendering 2D pose preview"):
        ok, frame = capture.read()
        if not ok:
            break
        for start, end in BODY18_EDGES:
            cv2.line(frame, tuple(body18_2d[frame_index, start].astype(int)), tuple(body18_2d[frame_index, end].astype(int)), (40, 220, 255), max(2, video_width // 500), cv2.LINE_AA)
        writer.write(frame)
    capture.release(); writer.release()
    add_source_audio(silent_2d_path, arguments.input_video, arguments.output_dir / "pose_2d_preview.mp4")

    silent_3d_path = arguments.output_dir / "pose_3d_preview_silent.mp4"
    writer = cv2.VideoWriter(str(silent_3d_path), cv2.VideoWriter_fourcc(*"mp4v"), frames_per_second, (960, 720))
    fused_preview_path = arguments.output_dir / "pose_3d_fused_preview_silent.mp4"
    fused_writer = cv2.VideoWriter(str(fused_preview_path), cv2.VideoWriter_fourcc(*"mp4v"), frames_per_second, (960, 720))
    for pose in fused_body18_3d:
        fused_writer.write(draw_numbered_3d_pose(pose))
    fused_writer.release()
    add_source_audio(fused_preview_path, arguments.input_video, arguments.output_dir / "pose_3d_fused_preview.mp4")

    for frame_index, pose in enumerate(tqdm(body18_3d, desc="Rendering 3D pose preview")):
        writer.write(draw_numbered_3d_pose(pose))
    writer.release()
    add_source_audio(silent_3d_path, arguments.input_video, arguments.output_dir / "pose_3d_preview.mp4")

    run_summary = {
        "frame_count": frame_count,
        "frames_per_second": frames_per_second,
        "pose_estimator": ({"adaptive": "OpenPose BODY18 + SDPose COCO17 adaptive fusion",
                            "sdpose-limbs": "SDPose COCO17 converted to BODY18",
                            "openpose": "CMU OpenPose BODY18 converted directly to PyTorch"})[arguments.pose_source],
        "pose_lifter": "MotionAGFormer-B trained on Human3.6M",
        "pose_postprocessing": "2D reprojection, fixed bone lengths, temporal smoothing, and foot grounding",
        "depth_estimator": depth_description,
        "vmd_converter": "VMD_3d_pose_baseline_multi original conversion functions",
        "pixels_per_unit": pixels_per_unit,
        "vmd_file": vmd_path.name,
        "fused_vmd_file": str(fused_vmd_path.relative_to(arguments.output_dir.resolve())),
        "foot_ik_enabled": not arguments.disable_foot_ik,
        "depth_standard_deviation": float(body18_3d[:, :, 2].std()),
        "short_pose_spikes_suppressed": spike_count,
        "short_pose_spike_policy": "Hampel-like local MAD; only runs up to 2 frames replaced",
        "flipped_depth_joints": flipped_depth_joints,
        "head_facing_camera": arguments.head_facing_camera,
    }
    (arguments.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8")


def main():
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    fps, width, height, frame_count = read_video_metadata(
        arguments.input_video, arguments.max_frames, arguments.start_frame)
    if arguments.stage in ("all", "pose"):
        estimate_2d_pose(arguments, frame_count)
    if arguments.stage in ("all", "depth"):
        estimate_depth_and_track_people(arguments, frame_count)
    if arguments.stage in ("all", "lift", "motion"):
        lift_pose_to_3d(arguments, width, height)
    if arguments.stage in ("all", "export"):
        export_results(arguments, fps, width, height, frame_count)


if __name__ == "__main__":
    main()
