"""Modern PyTorch pose lifting with the original 3d-pose-baseline contracts."""

from .motionagformer_backend import lift_openpose_json, lift_openpose_sequence

__all__ = ["lift_openpose_sequence", "lift_openpose_json"]
