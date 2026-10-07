"""SkyTNT ISNet inference; see vendor_runtime/anime_segmentation/LICENSE."""
from __future__ import annotations
import importlib.util
import threading
import time
from numbers import Integral, Real
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "assets" / "v181d"
_MODEL_CACHE = None
_MODEL_LOCK = threading.Lock()

def _get_model(device):
    global _MODEL_CACHE
    with _MODEL_LOCK:
        if _MODEL_CACHE is not None and _MODEL_CACHE[1] == device:
            return _MODEL_CACHE

        import torch
        from safetensors.torch import load_file

        source = ROOT / "vendor_runtime" / "anime_segmentation" / "isnet.py"
        if not source.is_file():
            raise RuntimeError(f"Missing official SkyTNT ISNet source: {source}")
        spec = importlib.util.spec_from_file_location("_whiteboard_skytnt_isnet", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # Avoid upstream model/__init__, Lightning and training imports.
        checkpoint = MODEL_ROOT / "model.safetensors"
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing bundled Anime Segmentation checkpoint: {checkpoint}")
        weights = load_file(str(checkpoint), device="cpu")
        state = {key.removeprefix("net."): value for key, value in weights.items()
                 if key.startswith("net.")}
        model = module.ISNetDIS()
        model.load_state_dict(state, strict=True)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select CPU explicitly.')
        model = model.float().eval().to(device)
        _MODEL_CACHE = model, device
        return _MODEL_CACHE


def release_subject_model():
    global _MODEL_CACHE
    with _MODEL_LOCK:
        _MODEL_CACHE = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect_subject_mask(image_rgb, resolution=1024, threshold=.5, *, device='cuda'):
    """Return all semantic foreground as an HxW bool mask, including detached parts.

    Foreground may contain multiple people; this model does not identify a main person.
    """
    if (not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8
            or not image_rgb.shape[0] or not image_rgb.shape[1]):
        raise ValueError("image_rgb must be a nonempty HxWx3 RGB uint8 array")
    if (isinstance(resolution, (bool, np.bool_)) or not isinstance(resolution, Integral)
            or resolution <= 0):
        raise ValueError("resolution must be a positive integer")
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be finite and between 0 and 1")

    import torch

    started = time.perf_counter()
    model, device = _get_model(device)
    h0, w0 = image_rgb.shape[:2]
    s = int(resolution)
    h, w = (s, max(1, int(s * w0 / h0))) if h0 > w0 else (max(1, int(s * h0 / w0)), s)
    top, left = (s - h) // 2, (s - w) // 2
    image = (image_rgb / 255).astype(np.float32)
    padded = np.zeros((s, s, 3), dtype=np.float32)
    padded[top:top + h, left:left + w] = cv2.resize(image, (w, h))
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)[None]).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        prediction = model(tensor)[0][0]
        if tuple(prediction.shape) != (1, 1, s, s) or not torch.isfinite(prediction).all().item():
            raise RuntimeError("Invalid ISNet prediction shape or nonfinite values")
        probabilities = prediction.sigmoid().float().cpu().numpy()[0, 0]
    probabilities = cv2.resize(probabilities[top:top + h, left:left + w], (w0, h0))
    if probabilities.shape != (h0, w0) or not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid restored ISNet prediction")
    mask = probabilities >= threshold
    return mask, {
        "detector": "SkyTNT Anime Segmentation (ISNet)",
        "device": device,
        "elapsed": time.perf_counter() - started,
        "coverage": float(mask.mean()),
        "foreground_scope": "Semantic foreground may include multiple people; detached components are retained.",
    }


