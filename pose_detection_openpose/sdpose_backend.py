"""SDPose/YOLO backend that preserves OpenPose COCO BODY_18 output files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EDGES = ((1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17))


class HeatmapHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(320, 320, 4, 2, 1, bias=False),
            torch.nn.InstanceNorm2d(320), torch.nn.SiLU())
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(320, 320, 1), torch.nn.InstanceNorm2d(320),
            torch.nn.SiLU())
        self.final_layer = torch.nn.Conv2d(320, 17, 1)

    def forward(self, value):
        return self.final_layer(self.conv_layers(self.deconv_layers(value)))

    @property
    def dtype(self):
        return self.final_layer.weight.dtype

    @property
    def device(self):
        return self.final_layer.weight.device


def _decode(heatmaps, box):
    heatmaps = heatmaps.float().cpu().numpy()[0]
    count, height, width = heatmaps.shape
    flat = heatmaps.reshape(count, -1)
    indexes = flat.argmax(1)
    yy, xx = np.divmod(indexes, width)
    scores = flat[np.arange(count), indexes]
    points = np.c_[xx, yy].astype(np.float32)
    for joint in range(count):
        x, y = int(xx[joint]), int(yy[joint])
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            points[joint, 0] += np.sign(heatmaps[joint, y, x + 1] - heatmaps[joint, y, x - 1]) * .25
            points[joint, 1] += np.sign(heatmaps[joint, y + 1, x] - heatmaps[joint, y - 1, x]) * .25
    x1, y1, x2, y2 = box
    points[:, 0] = x1 + (points[:, 0] + .5) / width * (x2 - x1)
    points[:, 1] = y1 + (points[:, 1] + .5) / height * (y2 - y1)
    return points, (1 / (1 + np.exp(-scores))).astype(np.float32)


def coco17_to_body18(points, confidence):
    body = np.zeros((18, 2), np.float32)
    scores = np.zeros(18, np.float32)
    mapping = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9,
               8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
               14: 2, 15: 1, 16: 4, 17: 3}
    for target, source in mapping.items():
        body[target], scores[target] = points[source], confidence[source]
    body[1] = (points[5] + points[6]) / 2
    scores[1] = min(confidence[5], confidence[6])
    return body, scores


def _load_pipeline(model_dir, device):
    sys.path.insert(0, str(ROOT / "SDPose_OOD"))
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from models.ModifiedUNet import Modified_forward
    from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
    from ultralytics import YOLO

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    unet = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", class_embed_type="projection",
        projection_class_embeddings_input_dim=4, torch_dtype=dtype,
        low_cpu_mem_usage=True)
    unet = Modified_forward(unet, "body")
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=dtype)
    head = HeatmapHead()
    head.load_state_dict(load_file(model_dir / "decoder/decoder.safetensors"), strict=True)
    pipe = SDPose_D_Pipeline(
        unet=unet, vae=vae, text_encoder=text, tokenizer=tokenizer,
        decoder=head,
        scheduler=DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler"))
    pipe.to(device, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe, YOLO(str(model_dir / "yolo11x.pt")), dtype


def _openpose_person(body, scores):
    packed = np.c_[body, scores].reshape(-1).astype(float).tolist()
    return {"person_id": [-1], "pose_keypoints_2d": packed,
            "face_keypoints_2d": [], "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [], "pose_keypoints_3d": [],
            "face_keypoints_3d": [], "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []}


def extract_openpose_keypoints(video_path, json_dir, rendered_video=None, *, model_dir=None,
                  max_people=1, start_frame=0, max_frames=None,
                  confidence_threshold=.25, cache_path=None, force=False):
    """Write OpenPose 1.5 compatible BODY_18 JSON and a rendered video."""
    video_path, json_dir = Path(video_path), Path(json_dir)
    model_dir = Path(model_dir or ROOT / "SDPose_body")
    json_dir.mkdir(parents=True, exist_ok=True)
    if cache_path and Path(cache_path).exists() and not force:
        return Path(cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe, detector, dtype = _load_pipeline(model_dir, device)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = min(total, start_frame + max_frames) if max_frames else total
    writer = None
    if rendered_video:
        rendered_video = Path(rendered_video)
        rendered_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(rendered_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    primary_points, primary_scores, primary_boxes = [], [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_index in tqdm(range(start_frame, stop), desc="Estimating BODY18 keypoints with SDPose"):
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.predict(frame, classes=[0], conf=confidence_threshold,
                                  imgsz=960, verbose=False,
                                  device=0 if device.type == "cuda" else "cpu")[0].boxes
        boxes = np.empty((0, 4), np.float32) if result is None else result.xyxy.cpu().numpy()
        if len(boxes):
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            boxes = boxes[np.argsort(-areas)[:max_people]]
        people, decoded = [], []
        for raw_box in boxes:
            x1, y1, x2, y2 = raw_box
            pad = .12 * max(x2 - x1, y2 - y1)
            box = (max(0, x1 - pad), max(0, y1 - pad),
                   min(width, x2 + pad), min(height, y2 + pad))
            bx1, by1, bx2, by2 = box
            crop = frame[int(by1):int(by2), int(bx1):int(bx2)]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(cv2.resize(crop, (768, 1024)), cv2.COLOR_BGR2RGB)
            value = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype) / 127.5 - 1
            context = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else torch.no_grad()
            with torch.inference_mode(), context:
                heatmaps = pipe(value, timesteps=[999], test_cfg={"flip_test": False}, show_progress_bar=False)
            coco, coco_scores = _decode(heatmaps, box)
            body, body_scores = coco17_to_body18(coco, coco_scores)
            people.append(_openpose_person(body, body_scores))
            decoded.append((coco, coco_scores, np.asarray(box, np.float32), body, body_scores))
        payload = {"version": 1.3, "people": people}
        name = f"{video_path.stem}_{frame_index:012d}_keypoints.json"
        (json_dir / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        if decoded:
            coco, scores, box, _, _ = decoded[0]
        else:
            coco, scores = np.zeros((17, 2), np.float32), np.zeros(17, np.float32)
            box = np.zeros(4, np.float32)
        primary_points.append(coco); primary_scores.append(scores); primary_boxes.append(box)
        if writer:
            for _, _, _, body, body_scores in decoded:
                for left, right in EDGES:
                    if body_scores[left] > .05 and body_scores[right] > .05:
                        cv2.line(frame, tuple(body[left].astype(int)), tuple(body[right].astype(int)), (40, 220, 255), 3, cv2.LINE_AA)
            writer.write(frame)
    cap.release()
    if writer:
        writer.release()
    cache_path = Path(cache_path or json_dir.parent / "sdpose_coco17.npz")
    np.savez_compressed(cache_path, keypoints=np.asarray(primary_points),
                        confidence=np.asarray(primary_scores), boxes=np.asarray(primary_boxes),
                        start_frame=np.int64(start_frame))
    del pipe, detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache_path


def extract_video(video_path, json_dir, rendered_video=None, *, model_dir=None,
                  number_people_max=1, frame_first=0, **kwargs):
    """Backward-compatible alias for :func:`extract_openpose_keypoints`."""
    return extract_openpose_keypoints(
        video_path, json_dir, rendered_video, model_dir=model_dir,
        max_people=number_people_max, start_frame=frame_first, **kwargs)
