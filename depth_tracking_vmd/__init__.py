"""Metric depth estimation and identity-aware multi-person tracking."""

from .unidepth_backend import estimate_depth_and_track_people, predict_video

__all__ = ["estimate_depth_and_track_people", "predict_video"]
