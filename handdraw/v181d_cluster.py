"""Full-visible Lab kmeans++ for .d, with an OpenCV reference/fallback."""
from __future__ import annotations

import cv2
import numpy as np


def capped_color_count(pixels, cap=16):
    seen = set()
    for begin in range(0, len(pixels), 4096):
        values = pixels[begin:begin + 4096].astype(np.uint32)
        packed = (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]
        for value in np.unique(packed):
            seen.add(int(value))
            if len(seen) >= cap:
                return cap
    return len(seen)


def choose_backend(backend="auto"):
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("cluster backend must be auto, cpu, or cuda.")
    if backend == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, OSError, RuntimeError) as exc:
        if backend == "cuda":
            raise RuntimeError("Torch CUDA clustering is unavailable.") from exc
    if backend == "cuda":
        raise RuntimeError("Torch CUDA clustering is unavailable.")
    return "cpu"


def cluster_backend(backend="auto"):
    return "torch-cuda" if choose_backend(backend) == "cuda" else "opencv-cpu"


def _bounded_chunk_size(pixels, count, requested, budget):
    available = min(budget, 6 * 1024 ** 3) - 48 * pixels - 64 * 1024 ** 2
    chunk = min(requested, 262144, pixels, available // (16 * max(count, 3)))
    if chunk < 1:
        raise RuntimeError("Full-resolution .d clustering exceeds its CUDA memory budget.")
    return int(chunk)


def _squared_distances(values, centers):
    # Direct FP32 channel arithmetic avoids TF32 and an N x K x 3 tensor.
    distance = (values[:, 0, None] - centers[None, :, 0]).square()
    for channel in (1, 2):
        distance.add_((values[:, channel, None] - centers[None, :, channel]).square())
    return distance


def _assign(pixels, centers, chunk_size):
    import torch
    labels = torch.empty(len(pixels), dtype=torch.int64, device=pixels.device)
    for begin in range(0, len(pixels), chunk_size):
        values = pixels[begin:begin + chunk_size]
        labels[begin:begin + len(values)] = _squared_distances(values, centers).argmin(dim=1)
    return labels


def _seed_centers(pixels, count, generator, chunk_size):
    import torch
    centers = torch.empty((count, 3), dtype=torch.float32, device=pixels.device)
    first = torch.randint(len(pixels), (), generator=generator, device=pixels.device)
    centers[0] = pixels[first]
    nearest = torch.empty(len(pixels), dtype=torch.float32, device=pixels.device)
    cumulative = torch.empty_like(nearest)
    for color in range(1, count):
        for begin in range(0, len(pixels), chunk_size):
            values = pixels[begin:begin + chunk_size]
            distances = _squared_distances(values, centers[color - 1:color]).flatten()
            if color > 1:
                distances = torch.minimum(nearest[begin:begin + len(values)], distances)
            nearest[begin:begin + len(values)] = distances
        torch.cumsum(nearest, dim=0, out=cumulative)
        draws = torch.rand(3, generator=generator, device=pixels.device) * cumulative[-1]
        candidates = torch.searchsorted(cumulative, draws, right=True).clamp_max(len(pixels) - 1)
        potential = torch.zeros(3, dtype=torch.float32, device=pixels.device)
        for begin in range(0, len(pixels), chunk_size):
            values = pixels[begin:begin + chunk_size]
            distances = _squared_distances(values, pixels[candidates])
            potential.add_(torch.minimum(distances, nearest[begin:begin + len(values), None]).sum(dim=0))
        centers[color] = pixels[candidates[potential.argmin()]]
    return centers


def _update_centers(pixels, labels, count, chunk_size):
    import torch
    sums = torch.zeros((count, 3), dtype=torch.float32, device=pixels.device)
    counts = torch.zeros(count, dtype=torch.int64, device=pixels.device)
    colors = torch.arange(count, device=pixels.device)
    # Fixed-order reductions avoid floating-point scatter atomics.
    for begin in range(0, len(pixels), chunk_size):
        values = pixels[begin:begin + chunk_size]
        membership = labels[begin:begin + len(values), None] == colors[None]
        counts.add_(membership.sum(dim=0))
        for channel in range(3):
            sums[:, channel].add_(torch.where(membership, values[:, channel, None], 0.).sum(dim=0))
    for empty in torch.nonzero(counts == 0).flatten().tolist():
        largest = counts.argmax()
        center = (sums[largest] / counts[largest]).reshape(1, 3)
        best = torch.tensor(-1., device=pixels.device)
        farthest = torch.tensor(0, dtype=torch.int64, device=pixels.device)
        for begin in range(0, len(pixels), chunk_size):
            values = pixels[begin:begin + chunk_size]
            distance = _squared_distances(values, center).flatten()
            distance.masked_fill_(labels[begin:begin + len(values)] != largest, -1.)
            maximum, index = distance.flip(0).max(dim=0)
            farthest = torch.where(maximum >= best, begin + len(values) - 1 - index, farthest)
            best = torch.maximum(best, maximum)
        point = pixels[farthest]
        labels[farthest] = empty
        counts[largest] -= 1
        counts[empty] = 1
        sums[largest] -= point
        sums[empty] = point
    return sums / counts[:, None]


def _torch_kmeans(samples, count, seed, device, chunk_size):
    import torch
    device = torch.device(device)
    budget = 6 * 1024 ** 3
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(budget, int(free * .8))
    chunk_size = _bounded_chunk_size(len(samples), count, chunk_size, budget)
    generator = torch.Generator(device=device).manual_seed(int(seed) & 0x7fffffff)
    scores, iterations = [], []
    best_score = float("inf")
    with torch.inference_mode():
        pixels = torch.as_tensor(samples, dtype=torch.float32, device=device)
        for _ in range(3):
            centers = _seed_centers(pixels, count, generator, chunk_size)
            labels = _assign(pixels, centers, chunk_size)
            for iteration in range(1, 10):
                previous = centers
                centers = _update_centers(pixels, labels, count, chunk_size)
                if iteration == 9 or bool((centers - previous).square().sum(dim=1).max() <= 1.):
                    break
                labels = _assign(pixels, centers, chunk_size)
            # Keep the last mean-update labels, including repaired empty clusters.
            objective = torch.zeros((), dtype=torch.float32, device=device)
            for begin in range(0, len(pixels), chunk_size):
                values = pixels[begin:begin + chunk_size]
                objective.add_((values - centers[labels[begin:begin + len(values)]]).square().sum())
            score = float(objective)
            scores.append(score)
            iterations.append(iteration + 1)
            if score < best_score:
                best_score = score
                best_labels, best_centers = labels.clone(), centers.clone()
        return (best_labels.to(device="cpu", dtype=torch.int32).numpy(), best_centers.cpu().numpy(),
                {"compactness": best_score, "attempt_compactness": scores, "iterations": iterations,
                 "chunk_size": chunk_size, "memory_budget_bytes": budget,
                 "working_set_bound_bytes": 48 * len(samples) + 64 * 1024 ** 2 + 16 * chunk_size * max(count, 3),
                 "torch_version": torch.__version__})


def cluster_lab(pixels, seed=42, *, backend="auto", chunk_size=262144, diagnostics=None, color_cap=16):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or pixels.shape[1] != 3 or not len(pixels):
        raise ValueError(".d clustering requires nonempty uint8 Lab pixels with shape (N, 3).")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("cluster chunk_size must be a positive integer.")
    selected = choose_backend(backend)
    selected_identity = "torch-cuda" if selected == "cuda" else "opencv-cpu"
    seed = int(seed) & 0x7fffffff
    count = capped_color_count(pixels, cap=color_cap)
    fallback = "torch_cuda_unavailable" if backend == "auto" and selected == "cpu" else ""
    details = {}
    if count == 1:
        labels, centers = np.zeros(len(pixels), np.int32), pixels[:1].astype(np.float32)
        details["compactness"] = 0.
        actual = selected_identity
    else:
        if selected == "cuda":
            try:
                labels, centers, details = _torch_kmeans(pixels, count, seed, "cuda", chunk_size)
            except RuntimeError as exc:
                if backend != "auto":
                    raise
                fallback = f"{type(exc).__name__}: {exc}"
                print(f".d CUDA clustering fallback: {fallback}", flush=True)
                selected = "cpu"
        if selected == "cpu":
            # The .d plan holds its existing OpenCV RNG lock through palette routing.
            cv2.setRNGSeed(seed)
            objective, labels, centers = cv2.kmeans(
                pixels.astype(np.float32), count, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.), 3, cv2.KMEANS_PP_CENTERS)
            labels = labels[:, 0]
            details["compactness"] = objective
        actual = "torch-cuda" if selected == "cuda" else "opencv-cpu"
    if diagnostics is not None:
        diagnostics.update(cluster_backend=actual, cluster_requested_backend=backend, cluster_seed=seed,
                           cluster_selected_backend=selected_identity,
                           cluster_fast_path="single-color" if count == 1 else "",
                           cluster_dtype="float32", cluster_pixels=len(pixels), cluster_colors=count,
                           cluster_attempts=3, cluster_max_iter=10, cluster_epsilon=1.,
                           cluster_initialization="kmeans++ / 3 D2 trials", cluster_centers=centers.tolist(),
                           cluster_fallback_reason=fallback,
                           **{f"cluster_{key}": value for key, value in details.items()})
    return labels, count
