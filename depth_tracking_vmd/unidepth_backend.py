"""UniDepth V2 inference with identity-aware multi-person tracking."""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(__file__).resolve().parent


@contextmanager
def _legacy_environment():
    """The original tracker resolves templates relative to its repository root."""
    old_cwd = Path.cwd()
    legacy_python = LEGACY_ROOT / "tensorflow"
    sys.path.insert(0, str(legacy_python))
    os.chdir(LEGACY_ROOT)
    try:
        yield legacy_python
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == str(legacy_python):
            sys.path.pop(0)


def _json_files(json_path):
    files = sorted(Path(json_path).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No OpenPose JSON files found in {json_path}")
    indexes = [int(re.findall(r"(\d{12})", item.name)[0]) for item in files]
    return files, indexes


def _sample_joints(depth, confidence, people, max_people):
    height, width = depth.shape
    person_depths, person_confidences, person_xy = [], [], []
    for person_index in range(max_people):
        packed = people[person_index]["pose_keypoints_2d"] if person_index < len(people) else [0.0] * 54
        values = np.asarray(packed, np.float32).reshape(18, 3)
        depths, confs, points = [], [], []
        for x, y, pose_confidence in values:
            if pose_confidence <= 0 or x < 0 or y < 0 or x >= width or y >= height:
                depths.append(0.0); confs.append(0.0); points.append([0, 0]); continue
            ix, iy = int(round(x)), int(round(y))
            x1, x2 = max(0, ix - 2), min(width, ix + 3)
            y1, y2 = max(0, iy - 2), min(height, iy + 3)
            patch, quality = depth[y1:y2, x1:x2], confidence[y1:y2, x1:x2]
            valid = np.isfinite(patch) & (patch > 0)
            depths.append(float(np.median(patch[valid])) if valid.any() else 0.0)
            confs.append(float(np.median(quality[valid]) * pose_confidence) if valid.any() else 0.0)
            points.append([ix, iy])
        person_depths.append(depths); person_confidences.append(confs); person_xy.append(points)
    return person_depths, person_confidences, person_xy


def _load_previous(path, frame_count, max_people):
    if not path:
        return None
    depth_file = Path(path) / "depth.txt"
    depth_z_file = Path(path) / "depth_z.txt"
    if not depth_file.exists() or not depth_z_file.exists():
        return None
    result = np.zeros((frame_count, max_people, 18), np.float32)
    people_seen = {}
    for row in csv.reader(depth_file.open(encoding="utf-8")):
        if not row:
            continue
        frame = int(row[0])
        values = np.asarray([float(value) for value in row[1:] if value.strip()], np.float32)
        person = people_seen.get(frame, 0)
        if 0 <= frame < frame_count and person < max_people and values.size >= 18:
            result[frame, person] = values[:18]
            people_seen[frame] = person + 1
    return result


def estimate_depth_and_track_people(video_path, json_path, output_path, *, model_path=None,
                  depth_interval=3, max_people=1, reverse_specific=None,
                  order_specific=None, past_depth_path=None, end_frame_no=-1,
                  verbose=1, resolution_level=0, force=False):
    """Run metric depth and retain legacy depth files and identity-sorted JSON.

    ``reverse_specific`` and ``order_specific`` use the dictionaries accepted by
    the original tracker. The output directories and text formats are unchanged.
    """
    video_path = Path(video_path).resolve()
    json_path = Path(json_path).resolve()
    output_path = Path(output_path).resolve()
    model_path = Path(model_path or ROOT / "unidepth-v2-vitl14").resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    raw_depth_file = output_path / "depth.txt"
    cache_file = output_path / "unidepth_metric.npz"
    if cache_file.exists() and raw_depth_file.exists() and not force:
        return {"depth_path": output_path, "cache_path": cache_file}
    files, frame_indexes = _json_files(json_path)
    start_frame, last_frame = frame_indexes[0], frame_indexes[-1]
    frame_count = last_frame - start_frame + 1
    if end_frame_no > 0:
        frame_count = min(frame_count, end_frame_no)
    interval = max(1, int(depth_interval))
    from unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained(model_path)
    model.resolution_level = resolution_level
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    absolute = np.zeros((frame_count, max_people, 18), np.float32)
    qualities = np.zeros_like(absolute)
    joint_pixel_coordinates = np.zeros((frame_count, max_people, 18, 2), np.int32)
    depth_maps = [np.empty(0, np.float32) for _ in range(frame_count)]
    previous = _load_previous(past_depth_path, frame_count, max_people)
    for local_index in tqdm(range(frame_count), desc="Estimating metric depth with UniDepth"):
        ok, frame = cap.read()
        if not ok:
            break
        infer = local_index % interval == 0 or local_index == frame_count - 1
        if not infer:
            continue
        payload_file = json_path / re.sub(r"\d{12}", f"{start_frame + local_index:012d}", files[0].name)
        people = json.loads(payload_file.read_text(encoding="utf-8")).get("people", []) if payload_file.exists() else []
        if previous is not None and np.any(previous[local_index]):
            absolute[local_index] = previous[local_index]
            continue
        rgb = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(device)
        with torch.inference_mode():
            prediction = model.infer(rgb)
        depth = prediction["depth"][0, 0].float().cpu().numpy()
        confidence = prediction["confidence"][0, 0].float().cpu().numpy()
        sampled, sampled_quality, sampled_xy = _sample_joints(depth, confidence, people, max_people)
        absolute[local_index], qualities[local_index], joint_pixel_coordinates[local_index] = sampled, sampled_quality, sampled_xy
        depth_maps[local_index] = depth[..., None]
    cap.release()
    with raw_depth_file.open("w", encoding="utf-8", newline="") as depth_handle, \
         (output_path / "depth_z.txt").open("w", encoding="utf-8", newline="") as z_handle:
        for index in range(frame_count):
            if index % interval == 0 or index == frame_count - 1:
                for person_index in range(max_people):
                    line = ",".join(str(float(value)) for value in absolute[index, person_index])
                    depth_handle.write(f"{index}, {line}\n")
                    z_handle.write(f"{index}, {line}\n")
    np.savez_compressed(cache_file, frames=np.arange(start_frame, start_frame + frame_count, interval),
                        depth_m=absolute[::interval, 0], confidence=qualities[::interval, 0])

    # Match the old FCRN normalization before handing data to its tracker.
    relative = absolute.copy()
    valid = relative[0, 0] > 0
    base_depth = float(relative[0, 0, valid].mean()) if valid.any() else 0.0
    for index in range(frame_count):
        if index % interval == 0 or index == frame_count - 1:
            relative[index] -= base_depth
    run_id = output_path.name.removesuffix("_depth")
    if force:
        for person_index in range(max_people):
            tracked = output_path / f"{json_path.name}_{run_id}_idx{person_index + 1:02d}"
            if tracked.exists():
                shutil.rmtree(tracked)
    cap_info = cv2.VideoCapture(str(video_path))
    source_width, source_height = int(cap_info.get(3)), int(cap_info.get(4))
    cap_info.release()
    for person_index in range(max_people):
        tracked = output_path / f"{json_path.name}_{run_id}_idx{person_index + 1:02d}"
        tracked.mkdir(parents=True, exist_ok=True)
        (tracked / "size.txt").write_text(f"{source_width}\n{source_height}\n", encoding="ascii")
    depth_images = output_path / "depth"
    if depth_images.exists() and force:
        shutil.rmtree(depth_images)
    depth_images.mkdir(exist_ok=True)
    reverse_specific, order_specific = reverse_specific or {}, order_specific or {}
    with _legacy_environment():
        import sort_people
        sorted_indexes = [[-1] * max_people for _ in range(frame_count)]
        frame_images = [[] for _ in range(interval)]
        max_confidence = [[0] * 18 for _ in range(max_people)]
        max_color_confidence = [[0] * 18 for _ in range(max_people)]
        past_data = [[] for _ in range(max_people)]
        past_depths = [[] for _ in range(max_people)]
        past_depths_z = [[] for _ in range(max_people)]
        pngs = []
        cap = cv2.VideoCapture(str(video_path)); cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        width, height = int(cap.get(3)), int(cap.get(4))
        for index in tqdm(range(frame_count), desc="Tracking person identities across frames"):
            ok, frame = cap.read()
            if not ok:
                break
            frame_images[index % interval] = np.asarray(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), dtype=np.float32)
            all_data, all_depths, all_depths_z = sort_people.sort(
                start_frame + index, start_frame + index, index, sorted_indexes,
                run_id, interval, str(depth_images), str(json_path), frame_count,
                max_people, reverse_specific, order_specific, files[0].name,
                start_frame, relative.tolist(), relative.tolist(), joint_pixel_coordinates.tolist(),
                depth_maps, frame_images, max_confidence, max_color_confidence,
                width, height, past_data, past_depths, past_depths_z, pngs, verbose)
            for person_index, source_index in enumerate(sorted_indexes[index]):
                past_data[source_index] = all_data[person_index]["people"][0]
                if index % interval == 0:
                    past_depths[source_index] = all_depths[person_index]
                    past_depths_z[source_index] = all_depths_z[person_index]
        cap.release()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.getLogger(__name__).info("Depth and tracked-person outputs written to %s", output_path)
    return {"depth_path": output_path, "cache_path": cache_file,
            "run_id": run_id, "start_frame": start_frame}


def predict_video(video_path, json_path, output_path, *, interval=3,
                  number_people_max=1, **kwargs):
    """Backward-compatible alias for the original adapter API."""
    return estimate_depth_and_track_people(
        video_path, json_path, output_path, depth_interval=interval,
        max_people=number_people_max, **kwargs)
