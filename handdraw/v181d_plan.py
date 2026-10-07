"""V1.8.1.d source routing, normalized independently of narration duration."""
from __future__ import annotations

from contextlib import nullcontext
import cv2
import numpy as np
from scipy import ndimage

from .grid import _grid_ink, _KMEANS_LOCK
from .v181d_cluster import cluster_backend, cluster_lab
from .v181d_subject_fill import build_subject_fill_plan

V181D_PLAN_VERSION = "v181d-subject-local-eased-torch-clean-v8-d-4"
V181E_PLAN_VERSION = "v181e-subject-native-background-clusters-1"


def build_v181d_ink_map(image_rgb, *, block_size=15, threshold_c=10, clahe_clip=2.,
                       bilateral_first=(7, 12, 3), bilateral_second=(5, 12, 2),
                       clahe_tiles=8, coarse_c=(5, 4), support_dilate=3,
                       component_area=4, component_span=8):
    """Approved native D: edge-preserving contrast and multiscale speck cleanup."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or not image_rgb.size:
        raise ValueError("A nonempty RGB image is required for D ink detection.")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, int(bilateral_first[0]), *bilateral_first[1:])
    gray = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(int(clahe_tiles), int(clahe_tiles))).apply(gray)
    gray = cv2.bilateralFilter(gray, int(bilateral_second[0]), *bilateral_second[1:])
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, block_size, threshold_c) == 0
    h, w = gray.shape
    support = np.zeros((h, w), bool)
    for factor, offset in [(0.5, coarse_c[0]), (1 / 3, coarse_c[1])]:
        reduced = cv2.resize(gray, (max(1, round(w * factor)), max(1, round(h * factor))),
                             interpolation=cv2.INTER_AREA)
        coarse = cv2.adaptiveThreshold(reduced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, offset) == 0
        restored = cv2.resize(coarse.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        support |= cv2.dilate(restored, np.ones((int(support_dilate), int(support_dilate)), np.uint8)) > 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats((ink & support).astype(np.uint8),
                                                         connectivity=8)
    scale = h / 720.0
    keep = (stats[:, cv2.CC_STAT_AREA] >= max(2, round(component_area * scale * scale))) | (
        np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= max(3, round(component_span * scale))
    )
    keep[0] = False
    return np.where(keep[labels], 0, 255).astype(np.uint8)


def _region_ids(image, ink, visible, seed=42, *, backend="auto", diagnostics=None, benchmark=None,
                palette_colors=16, min_region_area=80, barrier_size=3):
    with benchmark.measure("palette_preprocess") if benchmark is not None else nullcontext():
        lab = cv2.cvtColor(cv2.bilateralFilter(image, 9, 75, 75), cv2.COLOR_RGB2LAB)
    with benchmark.measure("palette_cluster", cuda=cluster_backend(backend) == "torch-cuda") if benchmark else nullcontext():
        labels, count = cluster_lab(lab[visible], seed, backend=backend, diagnostics=diagnostics, color_cap=palette_colors)
    with benchmark.measure("palette_components") if benchmark is not None else nullcontext():
        colors = np.full(ink.shape, -1, np.int32)
        colors[visible] = labels
        barrier = cv2.dilate(ink.astype(np.uint8), np.ones((int(barrier_size), int(barrier_size)), np.uint8)) > 0
        ids = np.zeros(ink.shape, np.int32)
        next_id = 1
        for color in range(count):
            components, total = ndimage.label((colors == color) & ~barrier)
            areas = np.bincount(components.ravel())
            kept = np.flatnonzero(areas >= min_region_area)
            kept = kept[kept > 0]
            lookup = np.zeros(total + 1, np.int32)
            lookup[kept] = np.arange(next_id, next_id + len(kept))
            selected = lookup[components]
            ids[selected > 0] = selected[selected > 0]
            next_id += len(kept)
    return ids


def build_v181d_plan(image_rgb, seed=42, *, subject_mask, visible_mask=None, cluster_backend="auto", benchmark=None,
                    background_clusters=False, grid_edge=10, clean_ink=True, ink_options=None,
                    palette_colors=16, min_region_area=80, barrier_size=3, fill_options=None, path_samples=4097):
    image, subject = np.asarray(image_rgb), np.asarray(subject_mask)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("V1.8.1.d requires a nonempty uint8 RGB image.")
    h, w = image.shape[:2]
    visible = np.ones((h, w), bool) if visible_mask is None else np.asarray(visible_mask, dtype=bool)
    if subject.shape != (h, w) or subject.dtype != np.bool_:
        raise ValueError("subject_mask must be a bool mask matching the image.")
    if visible.shape != (h, w) or not visible.any():
        raise ValueError("visible_mask must match the image and contain pixels.")
    with benchmark.measure("ink_grid") if benchmark else nullcontext():
        options = ink_options or {}
        ink, structure, ink_path = _grid_ink(image, visible, edge=grid_edge,
            block_size=options.get('block_size', 15), threshold_c=options.get('threshold_c', 10),
            ink_map=build_v181d_ink_map(image, **options) if clean_ink else None)
    ink_end = .45 if ink_path else 0.
    initial = np.asarray(ink_path[-1][1:] if ink_path else [(w - 1) / 2, (h - 1) / 2])
    info = {}
    with _KMEANS_LOCK:
        cv2.setRNGSeed(int(seed) & 0x7fffffff)
        ids = _region_ids(image, ink, visible, seed, backend=cluster_backend, diagnostics=info, benchmark=benchmark,
                          palette_colors=palette_colors, min_region_area=min_region_area, barrier_size=barrier_size)
        timing, _, regions, _, tips, lift = build_subject_fill_plan(
            image, ids, 0., 1., 30, initial, path_samples,
            subject_mask=subject, visible_mask=visible, diagnostics=info, benchmark=benchmark,
            **({"background_clusters": True} if background_clusters else {}), **(fill_options or {}))
    path = np.column_stack((ink_end + np.linspace(0, 1, len(tips)) * (1 - ink_end), tips)).astype(np.float32)
    if ink_path:
        before = np.asarray(ink_path, np.float32)
        before[:, 0] *= ink_end
        before = before[:-1] if len(before) > 1 else before
        path = np.vstack((before, path))
        lift = np.concatenate((np.zeros(len(before)), lift))
    color = np.full((h, w), 255, np.uint8)
    color[visible] = np.clip(np.round(timing[visible] * 254), 0, 254).astype(np.uint8)
    return {"structure_time": structure, "fill_time": np.full((h, w), 255, np.uint8),
            "color_reveal_time": color, "color_contact_time": timing,
            "pen_path": path, "pen_lift": lift.astype(np.float32), "ink_end": np.float32(ink_end),
            "metadata": {"version": V181E_PLAN_VERSION if background_clusters else V181D_PLAN_VERSION,
                         "grid_cells": len(ink_path),
                         "palette_regions": regions, "subject_pixels": int((subject & visible).sum()), **info}}
