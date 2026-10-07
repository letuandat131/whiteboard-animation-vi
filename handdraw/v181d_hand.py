"""Native photographic left-hand rigs with rigid markers and absolute-clock motion."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from stat import S_ISREG

import cv2
import numpy as np


HAND_RENDER_VERSION = "v181d-hand-long-turn-clock-v3"
_REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = {mode: _REPO_ROOT / "assets" / "v181d" / f"{mode}-hand-long.png"
          for mode in ("draw", "color")}
RIG_SETTINGS = {
    "reference_height": 1080, "native_scale": .76,
    "wrist_degrees": 11, "wrist_frequency": .55,
    "oscillations": [[1.3, 0], [1.05, 1.7], [1.45, 3.2], [.55, .4]],
    "marker_dilate": 17, "marker_falloff": 25, "nib_radius": 12,
    "shadow": [[.17, 18], [.07, 5]], "shadow_offset": [18, 22],
    "contact_opacity": .18, "contact_variance": [80, 40],
    "contact_offset": [6, 5.5], "shadow_max": .4,
    "lift_scale": .025, "lift_pixels": 12,
    "lift_shadow_offset": [8, 12], "lift_shadow_fade": .22,
    "poses": {
        "draw": {
            "native_size": [768, 2048], "native_scale": 1.18, "mirror": True, "tip": [125, 70],
            "marker": [[119, 62], [144, 66], [305, 205], [372, 287], [357, 309],
                       [335, 322], [316, 312], [276, 265], [145, 129]],
            "centers": [[263, 104, 56, 5, -4], [368, 187, 66, -4, -3],
                        [180, 200, 56, -3, 5], [397, 423, 108, 4, -1]],
        },
        "color": {
            "native_size": [735, 2140], "native_scale": 1.08, "mirror": False, "tip": [616, 102],
            "marker": [[585, 53], [629, 100], [638, 291], [625, 463], [610, 502],
                       [564, 494], [566, 434], [591, 264], [605, 137]],
            "centers": [[438, 168, 67, -4, 3], [533, 135, 74, 4, -3],
                        [633, 224, 70, -4, -4], [369, 572, 126, -4, -1]],
        },
    },
}


def _asset_stat(mode):
    if mode not in RIG_SETTINGS["poses"] or mode not in ASSETS:
        raise ValueError("Hand mode must be draw or color.")
    path = ASSETS[mode].resolve()
    if not path.is_relative_to(_REPO_ROOT):
        raise ValueError("Hand asset must be inside this repo.")
    stat = path.stat()
    if not S_ISREG(stat.st_mode):
        raise ValueError("Hand asset must be a regular PNG file.")
    return path, stat


def _asset_bytes(mode):
    path, _ = _asset_stat(mode)
    return path, path.read_bytes()


@lru_cache(maxsize=4)
def _asset_sha256(path, size, mtime_ns):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hand_asset_identity() -> dict:
    """JSON-safe fingerprint of both persisted assets and all rig settings."""
    assets = {}
    for mode in ASSETS:
        path, stat = _asset_stat(mode)
        assets[mode] = {"path": path.relative_to(_REPO_ROOT).as_posix(),
                        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "sha256": _asset_sha256(path, stat.st_size, stat.st_mtime_ns)}
    identity = {"version": HAND_RENDER_VERSION, "assets": assets,
                "settings": json.loads(json.dumps(RIG_SETTINGS))}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity["fingerprint"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return identity


def prepare_asset(mode, settings=None):
    _, encoded = _asset_bytes(mode)
    config = RIG_SETTINGS if settings is None else settings
    pose = config["poses"][mode]
    width, height = pose["native_size"]
    if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Hand asset must be a native uint8 RGBA PNG cutout.")
    raw = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
    if (raw is None or raw.dtype != np.uint8 or raw.shape != (height, width, 4)
            or not np.any(raw[..., 3] == 0) or not np.any(raw[..., 3] >= 200)):
        raise ValueError(f"Hand asset must be a transparent native uint8 RGBA {width}x{height} cutout.")
    tx, ty = pose["tip"]
    if raw[ty, tx, 3] < 128 or (settings is None and raw[ty, tx, :3].max() >= 128):
        raise ValueError("Hand asset must have a visible dark marker nib at its anchor.")
    polygon = np.array(pose["marker"], np.int32)
    if pose["mirror"]:
        raw = raw[:, ::-1]
        tx = width - 1 - tx
        polygon[:, 0] = width - 1 - polygon[:, 0]
    image = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255
    image[..., :3] *= image[..., 3:4]
    pen = np.zeros((height, width), np.uint8)
    cv2.fillPoly(pen, [polygon], 1)
    cv2.circle(pen, (tx, ty), int(config["nib_radius"]), 1, -1)
    edge = int(config["marker_dilate"])
    pen = cv2.dilate(pen, np.ones((edge, edge), np.uint8))
    return image, (tx, ty), pen


def motion_basis(mode, pen, settings=None):
    if mode not in RIG_SETTINGS["poses"]:
        raise ValueError("Hand mode must be draw or color.")
    config = RIG_SETTINGS if settings is None else settings
    pose = config["poses"][mode]
    width, height = pose["native_size"]
    if pen.shape != (height, width):
        raise ValueError("Marker mask must match the native hand dimensions.")
    y, x = np.indices(pen.shape, dtype=np.float32)
    if pose["mirror"]:
        x = width - 1 - x
    # Include a zero-motion guard around the nib, barrel and cap before feathering.
    distance = cv2.distanceTransform(1 - pen, cv2.DIST_L2, 3)
    protect = np.clip(distance / config["marker_falloff"], 0, 1)
    direction = -1 if pose["mirror"] else 1
    fields = []
    for cx, cy, radius, dx, dy in pose["centers"]:
        weight = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2)) * protect
        fields.append(np.stack((direction * dx * weight, dy * weight), axis=-1))
    return np.stack(fields, axis=-2)


def motion_at(seconds, settings=None):
    if not math.isfinite(seconds):
        raise ValueError("Hand motion seconds must be finite.")
    config = RIG_SETTINGS if settings is None else settings
    angle = math.radians(config["wrist_degrees"]) * math.sin(
        seconds * math.tau * config["wrist_frequency"])
    return (angle, *(math.sin(seconds * math.tau * frequency + phase)
                     for frequency, phase in config["oscillations"]))


def scheduled_hands(state, block, indices, panel, minimum_index, fps):
    """Return (batch slot, mode, page contact, absolute seconds, normalized lift)."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Hand scheduling fps must be finite and positive.")
    selected = [(slot, index) for slot, index in enumerate(indices)
                if max(0, minimum_index) <= index < block.draw_frames - 1]
    if not selected:
        return []
    path = np.asarray(state["pen_path"], dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 3 or not len(path)
            or not np.isfinite(path).all() or np.any(np.diff(path[:, 0]) <= 0)
            or np.any(path[:, 0] < 0) or np.any(path[:, 0] > 1)):
        raise ValueError("pen_path must be finite Nx3 with strictly increasing normalized times.")
    lift = np.asarray(state.get("pen_lift", np.zeros(len(path))), dtype=np.float64)
    if (lift.shape != (len(path),) or not np.isfinite(lift).all()
            or np.any(lift < 0) or np.any(lift > 1)):
        raise ValueError("pen_lift must be a normalized finite 1-D array matching pen_path.")
    ink_end = float(state["ink_end"])
    if not math.isfinite(ink_end) or not 0 <= ink_end <= 1:
        raise ValueError("ink_end must be finite and normalized.")
    px, py = float(panel["x"]), float(panel["y"])
    if not math.isfinite(px) or not math.isfinite(py):
        raise ValueError("Panel offsets must be finite.")
    events = []
    for slot, index in selected:
        progress = (index - max(0, minimum_index)) / max(1, block.draw_frames - max(0, minimum_index) - 1)
        point = (float(np.interp(progress, path[:, 0], path[:, 1])) + px,
                 float(np.interp(progress, path[:, 0], path[:, 2])) + py)
        events.append((slot, "draw" if progress < ink_end else "color", point,
                       (block.start_frame + index) / fps,
                       float(np.interp(progress, path[:, 0], lift))))
    return events


class MovingHand:
    def __init__(self, mode, width, height, device, settings=None):
        import torch
        if any(not isinstance(value, (int, np.integer)) or value < 1 for value in (width, height)):
            raise ValueError("Hand frame dimensions must be positive integers.")
        self.settings = RIG_SETTINGS if settings is None else settings
        image, self.tip, pen = prepare_asset(mode, self.settings)
        basis = motion_basis(mode, pen, self.settings)
        self.motion_margin = np.abs(basis).sum(axis=2).max(axis=(0, 1)) + 2
        self.native_height, self.native_width = image.shape[:2]
        self.width, self.height = width, height
        alpha = image[..., 3]
        shadow = sum(opacity * cv2.GaussianBlur(alpha, (0, 0), sigma)
                     for opacity, sigma in self.settings["shadow"])
        texture = np.concatenate((image, shadow[..., None]), axis=-1)
        self.texture = torch.from_numpy(texture).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        self.fields = torch.from_numpy(basis.reshape(self.native_height, self.native_width, -1))
        self.fields = self.fields.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        self.yy, self.xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                         torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        self.size = self.settings["poses"][mode]["native_scale"] * height / self.settings["reference_height"]
        ratio = height / self.settings["reference_height"]
        self.shadow_dx, self.shadow_dy = (max(1, round(offset * ratio))
                                         for offset in self.settings["shadow_offset"])
        self.contact_scale = max(.1, ratio) ** 2

    def stamp(self, frame, point, seconds, bounds, lift=0):
        import torch
        import torch.nn.functional as functional
        if (not torch.is_tensor(frame) or frame.dtype != torch.uint8
                or tuple(frame.shape) != (self.height, self.width, 3)
                or frame.device != self.texture.device):
            raise ValueError("Hand frame must be a matching HWC uint8 tensor on the rig device.")
        if len(point) != 2 or not all(math.isfinite(value) for value in (*point, seconds)):
            raise ValueError("Hand point and seconds must be finite.")
        if not math.isfinite(lift) or not 0 <= lift <= 1:
            raise ValueError("Hand lift must be finite and normalized.")
        if len(bounds) != 4 or any(not isinstance(value, (int, np.integer)) for value in bounds):
            raise ValueError("Hand bounds must be integer (left, top, right, bottom).")
        left, top, right, bottom = bounds
        if right < left or bottom < top:
            raise ValueError("Hand bounds must be ordered.")
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.width, right), min(self.height, bottom)
        if left >= right or top >= bottom:
            return frame
        config = self.settings
        ratio = self.height / config["reference_height"]
        size = self.size * (1 + config["lift_scale"] * lift)
        angle, *oscillations = motion_at(seconds, config)
        cosine, sine = math.cos(angle), math.sin(angle)
        extra_x, extra_y = config["lift_shadow_offset"]
        ox = self.shadow_dx + round(extra_x * ratio * lift)
        oy = self.shadow_dy + round(extra_y * ratio * lift)
        vx, vy = config["contact_variance"]
        cx, cy = config["contact_offset"]
        # Include deformation, bilinear support and both shadows before clipping to the canvas.
        mx, my = self.motion_margin
        corners = np.array([[-mx, -my], [self.native_width + mx, -my],
                            [-mx, self.native_height + my],
                            [self.native_width + mx, self.native_height + my]]) - self.tip
        bx = point[0] + size * (cosine * corners[:, 0] - sine * corners[:, 1])
        by = point[1] - config["lift_pixels"] * ratio * lift + size * (
            sine * corners[:, 0] + cosine * corners[:, 1])
        contact_x = math.sqrt(vx * self.contact_scale * 24)
        contact_y = math.sqrt(vy * self.contact_scale * 24)
        x0 = max(0, math.floor(min(bx.min(), point[0] + ox / cx - contact_x)))
        y0 = max(0, math.floor(min(by.min(), point[1] + oy / cy - contact_y)))
        x1 = min(self.width, math.ceil(max(bx.max() + ox, point[0] + ox / cx + contact_x)))
        y1 = min(self.height, math.ceil(max(by.max() + oy, point[1] + oy / cy + contact_y)))
        left, top, right, bottom = max(left, x0), max(top, y0), min(right, x1), min(bottom, y1)
        if x0 >= x1 or y0 >= y1:
            self.last_delta = (torch.zeros_like(self.xx), torch.zeros_like(self.yy))
            return frame
        xx, yy = self.xx[y0:y1, x0:x1], self.yy[y0:y1, x0:x1]
        dx = (xx - point[0]) / size
        dy = (yy - point[1] + config["lift_pixels"] * ratio * lift) / size
        sx = cosine * dx + sine * dy + self.tip[0]
        sy = -sine * dx + cosine * dy + self.tip[1]
        half_w, half_h = (self.native_width - 1) / 2, (self.native_height - 1) / 2
        grid = torch.stack((sx / half_w - 1, sy / half_h - 1), dim=-1).unsqueeze(0)
        local = functional.grid_sample(self.fields, grid, align_corners=True)[0]
        delta_x = sum(local[2 * i] * value for i, value in enumerate(oscillations))
        delta_y = sum(local[2 * i + 1] * value for i, value in enumerate(oscillations))
        padding = (x0, self.width - x1, y0, self.height - y1)
        self.last_delta = tuple(functional.pad(delta, padding) for delta in (delta_x, delta_y))
        if left >= right or top >= bottom:
            return frame
        grid[..., 0] -= delta_x / half_w
        grid[..., 1] -= delta_y / half_h
        sampled = functional.grid_sample(self.texture, grid, align_corners=True)[0]
        cast = torch.zeros_like(xx)
        if ox < x1 - x0 and oy < y1 - y0:
            cast[oy:, ox:] = sampled[4, :-oy, :-ox]
        contact = config["contact_opacity"] * (1 - lift) * torch.exp(
            -((xx - point[0] - ox / cx) ** 2 / (vx * self.contact_scale)
              + (yy - point[1] - oy / cy) ** 2 / (vy * self.contact_scale)))
        background = frame[y0:y1, x0:x1].permute(2, 0, 1).float() / 255
        shadow = (cast * (1 - config["lift_shadow_fade"] * lift) + contact).clamp(
            0, config["shadow_max"])
        background *= (1 - shadow).unsqueeze(0)
        rgb = background * (1 - sampled[3:4]) + sampled[:3]
        frame[top:bottom, left:right] = (rgb[:, top - y0:bottom - y0, left - x0:right - x0].permute(1, 2, 0)
                                       .clamp(0, 1) * 255).round().to(torch.uint8)
        return frame
