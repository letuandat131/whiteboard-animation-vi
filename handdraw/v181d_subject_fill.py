"""D-only local palette sweeps, with synchronized eased contact and pen-up lift."""
import math
from contextlib import nullcontext

import numpy as np
from skimage.measure import label

from .v181d_palette import _palette_regions


def _greedy_route(endpoints, initial_tip):
    remaining = np.ones(len(endpoints), bool)
    tip = initial_tip.copy()
    route = []
    while remaining.any():
        score = np.linalg.norm(endpoints - tip, axis=2) + .08 * np.maximum(tip[1] - endpoints[:, :, 1], 0)
        score[~remaining] = np.inf
        index, reverse = np.unravel_index(np.argmin(score), score.shape)
        remaining[index] = False
        route.append((int(index), int(reverse)))
        tip = endpoints[index, 1 - reverse]
    return route


def _route(endpoints, initial_tip, exit_points=None):
    """Compare greedy and three-choice lookahead, then refine orientations/2-opt."""
    n = len(endpoints)
    if not n:
        return []
    distance = np.linalg.norm(endpoints[:, None, :, None] -
                              endpoints[None, :, None, :], axis=-1)
    exit_cost = (np.linalg.norm(endpoints[:, :, None] - exit_points, axis=-1).min(axis=2)
                 if exit_points is not None and len(exit_points) else np.zeros((n, 2)))

    def orient(order):
        costs = np.linalg.norm(endpoints[order[0]] - initial_tip, axis=1)
        parents = []
        for previous, current in zip(order, order[1:]):
            options = costs[:, None] + distance[previous, current, ::-1, :]
            parents.append(options.argmin(axis=0))
            costs = options.min(axis=0)
        costs += exit_cost[order[-1], ::-1]
        reverse = int(costs.argmin())
        route = [(order[-1], reverse)]
        for index, parent in zip(order[-2::-1], parents[::-1]):
            reverse = int(parent[reverse])
            route.append((index, reverse))
        return route[::-1], float(costs.min())

    orders = []
    for lookahead in (False, True):
        remaining = np.ones(n, bool)
        tip = initial_tip.copy()
        order = []
        while remaining.any():
            costs = np.linalg.norm(endpoints - tip, axis=2)
            score = costs + .08 * np.maximum(tip[1] - endpoints[:, :, 1], 0)
            score[~remaining] = np.inf
            nearby = np.argsort(score.min(axis=1))[:min(3, int(remaining.sum()))]
            candidates = (nearby[:, None] * 2 + np.arange(2)).ravel()
            index, reverse = np.unravel_index(np.argmin(score), score.shape)
            if lookahead and remaining.sum() > 1:
                choices = []
                for flat in candidates:
                    first, direction = np.unravel_index(flat, score.shape)
                    next_cost = distance[first, :, 1 - direction, :].copy()
                    next_cost[~remaining] = np.inf
                    next_cost[first] = np.inf
                    nearby_next = np.argsort(next_cost.min(axis=1))[:min(3, int(remaining.sum()) - 1)]
                    next_candidates = (nearby_next[:, None] * 2 + np.arange(2)).ravel()
                    best = np.inf
                    for next_flat in next_candidates:
                        second, second_direction = np.unravel_index(next_flat, next_cost.shape)
                        if remaining.sum() > 2:
                            final_cost = distance[second, :, 1 - second_direction, :].copy()
                            final_cost[~remaining] = np.inf
                            final_cost[[first, second]] = np.inf
                            tail = float(final_cost.min())
                        else:
                            tail = exit_cost[second, 1 - second_direction]
                        best = min(best, next_cost[second, second_direction] + tail)
                    choices.append((costs[first, direction] + best, score[first, direction],
                                    int(first), int(direction)))
                _, _, index, reverse = min(choices)
            order.append(int(index))
            remaining[index] = False
            tip = endpoints[index, 1 - reverse]
        orders.append(orient(order))
    route, cost = min(orders, key=lambda item: item[1])
    # Reversing a block and its stroke directions preserves internal transitions.
    # Bound both the search window and improvement passes.
    for _ in range(2):
        best, change = 0., None
        for left in range(n):
            previous = initial_tip if left == 0 else endpoints[route[left - 1][0], 1 - route[left - 1][1]]
            first, first_direction = route[left]
            for right in range(left + 1, min(n, left + 13)):
                last, last_direction = route[right]
                old = np.linalg.norm(previous - endpoints[first, first_direction])
                new = np.linalg.norm(previous - endpoints[last, 1 - last_direction])
                if right + 1 < n:
                    after, after_direction = route[right + 1]
                    old += distance[last, after, 1 - last_direction, after_direction]
                    new += distance[first, after, first_direction, after_direction]
                else:
                    old += exit_cost[last, 1 - last_direction]
                    new += exit_cost[first, first_direction]
                if old - new > best + 1e-9:
                    best, change = float(old - new), (left, right)
        if change is None:
            break
        left, right = change
        route[left:right + 1] = [(index, 1 - reverse) for index, reverse in route[left:right + 1][::-1]]
        route, cost = orient([index for index, _ in route])
    return route


def _compact_route(endpoints, order, initial_tip, boundaries):
    """Orient across colors and reverse bounded blocks within each color only."""
    n = len(order)
    if not n:
        return []
    distance = np.linalg.norm(endpoints[:, None, :, None] -
                              endpoints[None, :, None, :], axis=-1)
    entry = np.linalg.norm(endpoints - initial_tip, axis=2)

    def orient(order):
        costs = entry[order[0]].copy()
        parents = []
        for previous, current in zip(order, order[1:]):
            options = costs[:, None] + distance[previous, current, ::-1, :]
            parents.append(options.argmin(axis=0))
            costs = options.min(axis=0)
        reverse = int(costs.argmin())
        route = [(order[-1], reverse)]
        for index, parent in zip(order[-2::-1], parents[::-1]):
            reverse = int(parent[reverse])
            route.append((index, reverse))
        return route[::-1]

    route = orient(order)
    for _ in range(12):
        best, change = 0., None
        for low, high in boundaries:
            for left in range(low, high):
                first, first_direction = route[left]
                if left:
                    before, before_direction = route[left - 1]
                    entering = distance[before, :, 1 - before_direction, :]
                else:
                    entering = entry
                for right in range(left + 1, min(high, left + 25)):
                    last, last_direction = route[right]
                    old = entering[first, first_direction]
                    new = entering[last, 1 - last_direction]
                    if right + 1 < n:
                        after, after_direction = route[right + 1]
                        old += distance[last, after, 1 - last_direction, after_direction]
                        new += distance[first, after, first_direction, after_direction]
                    if old - new > best + 1e-9:
                        best, change = float(old - new), (left, right)
        if change is None:
            break
        left, right = change
        order = [index for index, _ in route]
        order[left:right + 1] = order[left:right + 1][::-1]
        route = orient(order)
    return route


def _merge_short(pending, u, v, brush_width, cosine, sine):
    """Combine only short neighboring spans that fit the original brush corridor."""
    active = list(pending)
    for index in range(len(active)):
        if active[index] is None:
            continue
        selected, begin, end, v0, v1 = active[index]
        if v1 - v0 > brush_width * .65:
            continue
        for other in range(index + 1, len(active)):
            item = active[other]
            if item is None or item[4] - item[3] > brush_width * .65:
                continue
            low, high = min(v0, item[3]), max(v1, item[4])
            gap = max(v0 - item[4], item[3] - v1, 0.)
            endpoint_gap = np.linalg.norm(np.array([begin, end])[:, None] - np.array([item[1], item[2]]), axis=2).min()
            if gap > brush_width * .75 or high - low > brush_width * 1.5 or endpoint_gap > brush_width:
                continue
            combined = np.concatenate((selected, item[0]))
            low_u, high_u = float(u[combined].min()), float(u[combined].max())
            if high_u - low_u > brush_width:
                continue
            lane_u = (low_u + high_u) * .5
            begin = np.array([lane_u * cosine - low * sine, lane_u * sine + low * cosine])
            end = np.array([lane_u * cosine - high * sine, lane_u * sine + high * cosine])
            selected, v0, v1 = combined, low, high
            active[index] = (selected, begin, end, v0, v1)
            active[other] = None
            if v1 - v0 > brush_width * .65:
                break
    return [item for item in active if item is not None]


def build_subject_fill_plan(image, region_ids, start, duration, fps, initial_tip, frame_count,
                         *, subject_mask=None, visible_mask=None, diagnostics=None, benchmark=None,
                         background_clusters=False, brush_min=24., brush_max=92., brush_ratio=.12,
                         stroke_angle=12., travel_speed_ratio=1.5):
    with benchmark.measure("stroke_groups") if benchmark is not None else nullcontext():
        palette, patches = _palette_regions(image, region_ids, visible_mask=visible_mask)
        h, w = palette.shape
        visible = np.ones((h, w), bool) if visible_mask is None else np.asarray(visible_mask, dtype=bool)
        if visible.shape != (h, w) or not visible.any():
            raise ValueError("visible_mask must match the image and contain pixels.")
        if subject_mask is not None:
            subject_mask = np.asarray(subject_mask)
            if subject_mask.shape != (h, w) or subject_mask.dtype != np.bool_:
                raise ValueError("subject_mask must be a bool mask matching the image.")
            patches = label(palette + (~subject_mask) * int(palette.max()),
                            connectivity=2).astype(np.int32)
        patches[~visible] = 0
        if duration <= 0 or frame_count < 1 or not np.isfinite([start, duration]).all():
            raise ValueError("duration and frame_count must be positive and times finite.")
        tip = np.asarray(initial_tip, float)
        if tip.shape != (2,) or not np.isfinite(tip).all():
            raise ValueError("initial_tip must be a finite pair.")
        brush_width = max(brush_min, min(brush_max, min(h, w) * brush_ratio))
        cosine, sine = math.cos(math.radians(stroke_angle)), math.sin(math.radians(stroke_angle))
        groups = []
        phases = [subject_mask & visible, ~subject_mask & visible] if subject_mask is not None else [visible]
        for phase_index, phase in enumerate(phases):
            ys, xs = np.where(phase)
            if not len(ys):
                continue
            # Choose the first substantial occurrence of each color from the top.
            colors = sorted(np.unique(palette[phase]), key=lambda color: (
                np.percentile(ys[palette[ys, xs] == color], 5), int(color)))
            for color in colors:
                ys, xs = np.where(phase & (palette == color))
                u, v = xs * cosine + ys * sine, -xs * sine + ys * cosine
                lanes = np.floor((u - u.min()) / brush_width).astype(np.int32)
                pending = []
                # A shared lane grid merges nearby fragmented patches of this color.
                # Only short empty gaps can be swept through; distant islands split.
                for lane in np.unique(lanes):
                    indexes = np.flatnonzero(lanes == lane)
                    order = indexes[np.argsort(v[indexes])]
                    breaks = np.flatnonzero(np.diff(v[order]) > brush_width * .75) + 1
                    for selected in np.split(order, breaks):
                        lane_u = (u[selected].min() + u[selected].max()) * .5
                        v0, v1 = float(v[selected[0]]), float(v[selected[-1]])
                        begin = np.array([lane_u * cosine - v0 * sine,
                                          lane_u * sine + v0 * cosine])
                        end = np.array([lane_u * cosine - v1 * sine,
                                        lane_u * sine + v1 * cosine])
                        pending.append((selected, begin, end, v0, v1))
                pending = _merge_short(pending, u, v, brush_width, cosine, sine)
                groups.append((phase_index, ys, xs, v, pending))
    with benchmark.measure("route_order") if benchmark is not None else nullcontext():
        candidates = []
        # Check actual complete travel: a nearest next-color endpoint is only a hint.
        for mode in ("greedy", "local", "exit"):
            tip = np.asarray(initial_tip, float).copy()
            strokes = []
            for group_index, (phase_index, ys, xs, v, pending) in enumerate(groups):
                endpoints = np.array([[item[1], item[2]] for item in pending])
                exit_points = (np.array([[item[1], item[2]] for item in groups[group_index + 1][4]]).reshape(-1, 2)
                               if mode == "exit" and group_index + 1 < len(groups) else None)
                route = _greedy_route(endpoints, tip) if mode == "greedy" else _route(endpoints, tip, exit_points)
                for index, reverse in route:
                    selected, begin, end, v0, v1 = pending[index]
                    if reverse:
                        begin, end, v0, v1 = end, begin, v1, v0
                    travel = float(np.linalg.norm(begin - tip))
                    length = max(abs(v1 - v0), brush_width * .35)
                    progress = ((v[selected] - v0) / (v1 - v0) if abs(v1 - v0) > 1e-6
                                else np.full(len(selected), .5))
                    strokes.append((ys[selected], xs[selected], progress, begin, end,
                                    travel, length, tip.copy()))
                    tip = end
            candidates.append(strokes)
        boundaries, offset = [], 0
        for group in groups:
            count = len(group[4])
            boundaries.append((offset, offset + count))
            offset += count
        for original in candidates[:]:
            endpoints = np.array([[stroke[3], stroke[4]] for stroke in original])
            route = _compact_route(endpoints, list(range(len(original))),
                                   np.asarray(initial_tip, float), boundaries)
            tip = np.asarray(initial_tip, float).copy()
            refined = []
            for index, reverse in route:
                ys, xs, progress, begin, end, _, length, _ = original[index]
                if reverse:
                    begin, end, progress = end, begin, 1 - progress
                travel = float(np.linalg.norm(begin - tip))
                refined.append((ys, xs, progress, begin, end, travel, length, tip.copy()))
                tip = end
            candidates.append(refined)
        strokes = min(candidates, key=lambda candidate: sum(stroke[5] for stroke in candidate))
        if diagnostics is not None:
            diagnostics.update(brush_strokes=len(strokes),
                               travel_pixels=sum(stroke[5] for stroke in strokes),
                               greedy_travel_pixels=sum(stroke[5] for stroke in candidates[0]))

    with benchmark.measure("contact_timeline") if benchmark is not None else nullcontext():
        # Both nib position and opaque pixel contact use this one eased timeline.
        weights = np.array([[max(stroke[5] / travel_speed_ratio, brush_width * .04), stroke[6]]
                            for stroke in strokes], float)
        durations = duration * weights / weights.sum()
        timing = np.full((h, w), np.inf, np.float32)
        frame_times = np.linspace(start, start + duration, frame_count)
        tips = np.broadcast_to(np.asarray(initial_tip, float), (frame_count, 2)).copy()
        lift = np.zeros(frame_count, float)
        current = start
        for stroke, (travel_duration, paint_duration) in zip(strokes, durations):
            ys, xs, progress, begin, end, travel, length, previous = stroke
            paint_start = current + travel_duration
            paint_end = paint_start + paint_duration
            travel_frames = (frame_times >= current) & (frame_times < paint_start)
            q = np.clip((frame_times[travel_frames] - current) / travel_duration, 0, 1)
            ease = .5 - .5 * np.cos(np.pi * q)
            tips[travel_frames] = previous + ease[:, None] * (begin - previous)
            lift[travel_frames] = np.sin(np.pi * q)
            paint_frames = (frame_times >= paint_start) & (frame_times <= paint_end + 1e-9)
            q = np.clip((frame_times[paint_frames] - paint_start) / paint_duration, 0, 1)
            ease = .5 - .5 * np.cos(np.pi * q)
            tips[paint_frames] = begin + ease[:, None] * (end - begin)
            # Invert position easing: contact occurs exactly when the nib reaches v.
            contact = np.arccos(1 - 2 * np.clip(progress, 0, 1)) / np.pi
            timing[ys, xs] = paint_start + contact * paint_duration
            current = paint_end
        end_time = np.float32(start + duration)
        if float(end_time) > start + duration:
            end_time = np.nextafter(end_time, np.float32(-np.inf))
        np.minimum(timing, end_time, out=timing)
        timing[~visible] = np.inf
        tips[:, 0] = np.clip(tips[:, 0], 0, w - 1)
        tips[:, 1] = np.clip(tips[:, 1], 0, h - 1)
    plan = timing, patches, int(patches.max()), 0.0, tips, lift
    if background_clusters:
        from .v181e_background_fill import reschedule_background
        with benchmark.measure("background_reschedule") if benchmark is not None else nullcontext():
            plan = reschedule_background(plan, strokes, durations, region_ids, subject_mask,
                                         start, duration, initial_tip, brush_width)
            plan[0][~visible] = np.inf
    return plan
