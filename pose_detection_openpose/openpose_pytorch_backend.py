"""PyTorch OpenPose BODY18 backend with OpenPose-compatible JSON output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OPENPOSE_ROOT = ROOT / "OpenPose_PyTorch"
EDGES = ((1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17))


def _person_from_openpose(candidate, row):
    keypoints = np.zeros((18, 3), np.float32)
    for joint in range(18):
        candidate_index = int(row[joint])
        if candidate_index >= 0:
            keypoints[joint] = candidate[candidate_index, :3]
    return {
        "person_id": [-1],
        "pose_keypoints_2d": keypoints.reshape(-1).astype(float).tolist(),
        "face_keypoints_2d": [], "hand_left_keypoints_2d": [],
        "hand_right_keypoints_2d": [], "pose_keypoints_3d": [],
        "face_keypoints_3d": [], "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": [],
    }, keypoints


def extract_openpose_keypoints(video_path, json_dir, rendered_video=None, *, model_dir=None,
                               max_people=1, start_frame=0, max_frames=None,
                               cache_path=None, force=False, **_):
    """Run the converted CMU OpenPose network and write native BODY18 JSON."""
    video_path, json_dir = Path(video_path), Path(json_dir)
    cache_path = Path(cache_path or json_dir.parent / "openpose_body18.npz")
    if cache_path.exists() and not force:
        return cache_path
    model_path = Path(model_dir or OPENPOSE_ROOT / "model") / "body_pose_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch OpenPose weights not found: {model_path}")
    json_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OPENPOSE_ROOT))
    from src.body import Body

    estimator = Body(str(model_path))
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(3)), int(capture.get(4))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = min(total, start_frame + max_frames) if max_frames else total
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = None
    if rendered_video:
        rendered_video = Path(rendered_video)
        rendered_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(rendered_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

    all_keypoints, all_confidence = [], []
    for frame_index in tqdm(range(start_frame, stop), desc="Estimating native OpenPose BODY18 keypoints"):
        ok, frame = capture.read()
        if not ok:
            break
        inference_scale = min(1.0, 1024.0 / height)
        inference_frame = (frame if inference_scale == 1.0 else cv2.resize(
            frame, None, fx=inference_scale, fy=inference_scale, interpolation=cv2.INTER_AREA))
        candidate, subset = estimator(inference_frame)
        if len(candidate) and inference_scale != 1.0:
            candidate[:, :2] /= inference_scale
        if len(subset):
            subset = subset[np.argsort(-subset[:, -2])[:max_people]]
        people, frame_people = [], []
        for row in subset:
            person, keypoints = _person_from_openpose(candidate, row)
            people.append(person); frame_people.append(keypoints)
        payload = {"version": 1.3, "people": people}
        output_name = f"{video_path.stem}_{frame_index:012d}_keypoints.json"
        (json_dir / output_name).write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8")

        primary = frame_people[0] if frame_people else np.zeros((18, 3), np.float32)
        all_keypoints.append(primary[:, :2]); all_confidence.append(primary[:, 2])
        if writer:
            for keypoints in frame_people:
                for left, right in EDGES:
                    if keypoints[left, 2] > 0 and keypoints[right, 2] > 0:
                        cv2.line(frame, tuple(keypoints[left, :2].astype(int)),
                                 tuple(keypoints[right, :2].astype(int)),
                                 (40, 220, 255), 3, cv2.LINE_AA)
            writer.write(frame)

    capture.release()
    if writer:
        writer.release()
    np.savez_compressed(
        cache_path, body18=np.asarray(all_keypoints), confidence=np.asarray(all_confidence),
        start_frame=np.int64(start_frame))
    del estimator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache_path


def extract_video(video_path, json_dir, rendered_video=None, *, model_dir=None,
                  number_people_max=1, frame_first=0, **kwargs):
    return extract_openpose_keypoints(
        video_path, json_dir, rendered_video, model_dir=model_dir,
        max_people=number_people_max, start_frame=frame_first, **kwargs)
