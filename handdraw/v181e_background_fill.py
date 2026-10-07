"""E-only background tail: finish edge-bounded clusters from left to right."""
import numpy as np
from scipy import ndimage
from skimage.measure import label

from .v181d_subject_fill import _greedy_route


def _order_background(strokes, region_ids, foreground, initial_tip, brush_width):
    # Keep raw edge IDs: palette IDs can merge neighboring regions of the same color.
    ids = np.asarray(region_ids, np.int32).copy()
    if not np.any(ids > 0):
        ids[:] = 1
    elif np.any(ids == 0):
        nearest = ndimage.distance_transform_edt(ids == 0, return_distances=False, return_indices=True)
        ids = ids[nearest[0], nearest[1]]
    clusters = label(np.where(foreground, 0, ids), connectivity=2).astype(np.int32)
    groups = {}
    for ys, xs, progress, begin, end, _, _, _ in strokes:
        labels = clusters[ys, xs]
        for rid in np.unique(labels):
            selected = labels == rid
            p = progress[selected]
            low, high = float(p.min()), float(p.max())
            a = begin + low * (end - begin)
            b = begin + high * (end - begin)
            contact = (p - low) / (high - low) if high - low > 1e-12 else np.full(len(p), .5)
            length = max(float(np.linalg.norm(b - a)), brush_width * .35)
            groups.setdefault(int(rid), []).append((ys[selected], xs[selected], contact, a, b, 0., length, a))
    centers = {rid: sum(float(s[1].sum()) for s in group) / sum(len(s[1]) for s in group)
               for rid, group in groups.items()}
    ordered = []
    tip = np.asarray(initial_tip, float)
    for rid in sorted(groups, key=lambda rid: (centers[rid], rid)):
        group = groups[rid]
        endpoints = np.array([[s[3], s[4]] for s in group])
        for index, reverse in _greedy_route(endpoints, tip):
            ys, xs, progress, begin, end, _, length, _ = group[index]
            if reverse:
                begin, end, progress = end, begin, 1 - progress
            ordered.append((ys, xs, progress, begin, end,
                            float(np.linalg.norm(begin - tip)), length, tip.copy()))
            tip = end
    return ordered


def reschedule_background(plan, strokes, native_durations, region_ids, subject_mask,
                          start, duration, initial_tip, brush_width):
    foreground = subject_mask if subject_mask is not None else np.zeros(region_ids.shape, bool)
    count = sum(bool(foreground[s[0][0], s[1][0]]) for s in strokes)
    background_start = float(start + native_durations[:count].sum())
    current_tip = strokes[count - 1][4].copy() if count else np.asarray(initial_tip, float)
    background = _order_background(strokes[count:], region_ids, foreground, current_tip, brush_width)
    if not background:
        return plan

    # The native subject timeline is already final. Only overwrite the background tail.
    timing, patches, regions, fade, tips, lift = plan
    ordered = []
    for ys, xs, progress, begin, end, _, length, _ in background:
        if np.linalg.norm(end - current_tip) < np.linalg.norm(begin - current_tip):
            begin, end, progress = end, begin, 1 - progress
        travel = float(np.linalg.norm(begin - current_tip))
        ordered.append((ys, xs, progress, begin, end, travel, length, current_tip.copy()))
        current_tip = end
    weights = np.array([[max(s[5] / 1.5, brush_width * .04), s[6]] for s in ordered])
    durations = (start + duration - background_start) * weights / weights.sum()
    frame_times = np.linspace(start, start + duration, len(tips))
    current = background_start
    for stroke, (travel_duration, paint_duration) in zip(ordered, durations):
        ys, xs, progress, begin, end, _, _, previous = stroke
        paint_start = current + travel_duration
        paint_end = paint_start + paint_duration
        travelling = (frame_times >= current) & (frame_times < paint_start)
        q = np.clip((frame_times[travelling] - current) / travel_duration, 0, 1)
        tips[travelling] = previous + (.5 - .5 * np.cos(np.pi * q))[:, None] * (begin - previous)
        lift[travelling] = np.sin(np.pi * q)
        painting = (frame_times >= paint_start) & (frame_times <= paint_end + 1e-9)
        q = np.clip((frame_times[painting] - paint_start) / paint_duration, 0, 1)
        tips[painting] = begin + (.5 - .5 * np.cos(np.pi * q))[:, None] * (end - begin)
        lift[painting] = 0
        contact = np.arccos(1 - 2 * np.clip(progress, 0, 1)) / np.pi
        timing[ys, xs] = paint_start + contact * paint_duration
        current = paint_end
    end_time = np.float32(start + duration)
    if float(end_time) > start + duration:
        end_time = np.nextafter(end_time, np.float32(-np.inf))
    timing[~foreground] = np.minimum(timing[~foreground], end_time)
    tips[:, 0] = np.clip(tips[:, 0], 0, timing.shape[1] - 1)
    tips[:, 1] = np.clip(tips[:, 1], 0, timing.shape[0] - 1)
    return timing, patches, regions, fade, tips, lift
