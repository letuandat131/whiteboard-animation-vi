"""Native grid routing extracted from srt-whiteboard; see vendor license."""
from __future__ import annotations
from typing import Sequence
import math
import cv2
import numpy as np

def _to_grid_blocks(image: np.ndarray, edge: int) -> np.ndarray:
    """把 HxW（x C）图像切成 (行数, 列数, edge, edge[, C]) 的分块视图。"""
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    if h % edge or w % edge:
        raise ValueError(f"图像尺寸 {w}x{h} 必须是 {edge} 的整数倍")
    rows, cols = h // edge, w // edge
    if image.ndim == 2:
        return image.reshape(rows, edge, cols, edge).transpose(0, 2, 1, 3)
    return image.reshape(rows, edge, cols, edge, image.shape[2]).transpose(0, 2, 1, 3, 4)


def _active_mask(threshold_map: np.ndarray, edge: int, threshold: int) -> np.ndarray:
    """哪些网格含墨迹：块内存在灰度低于阈值的像素即为真。"""
    blocks = _to_grid_blocks(threshold_map, edge)
    return np.any(blocks < threshold, axis=(2, 3))


def _label_components(active: np.ndarray) -> tuple[np.ndarray, int]:
    """对墨迹格做 8 连通连通域标记，返回 (标签图, 域数)。"""
    n, labels = cv2.connectedComponents(active.astype(np.uint8), connectivity=8)
    return labels, n - 1  # 去掉背景标签 0


def _component_cells(labels: np.ndarray, label: int) -> list[tuple[int, int]]:
    coords = np.argwhere(labels == label)
    return [(int(r), int(c)) for r, c in coords]


def _merge_small_components(
    components: list[list[tuple[int, int]]],
    merge_threshold: int,
) -> list[list[tuple[int, int]]]:
    """
    把小连通域（格数 ≤ merge_threshold）合并到空间最近的大连通域。
    避免大量 1-2 格的碎片穿插在大块文字之间，导致“画一块字没画完就跳走”。
    若没有大连通域可并入，则保留原样（不丢弃任何墨迹）。
    """
    if not components:
        return components
    big = [c for c in components if len(c) > merge_threshold]
    small = [c for c in components if len(c) <= merge_threshold]
    if not small or not big:
        return components

    # 预算每个大区域的质心
    centroids = []
    for cells in big:
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        centroids.append((sum(rs) / len(rs), sum(cs) / len(cs)))

    # 每个小碎片并入最近的大区域
    merged = [list(cells) for cells in big]  # 拷贝，可追加
    for cells in small:
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        cr = sum(rs) / len(rs)
        cc = sum(cs) / len(cs)
        best = min(
            range(len(big)),
            key=lambda i: (centroids[i][0] - cr) ** 2 + (centroids[i][1] - cc) ** 2,
        )
        merged[best].extend(cells)
    return merged


def _bounds(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _split_bridge_connected_component(
    cells: list[tuple[int, int]],
    min_side_cells: int = 20,
) -> list[list[tuple[int, int]]]:
    """Split a very wide component when it is connected only by a thin bridge.

    A baseline, arrow, or stray outline can join separate objects into one
    connected component.  Drawing that component with one nearest-neighbour
    walk makes the pen alternate between those objects.  Valleys in the
    vertical ink projection are reliable weak-bridge signals at grid scale.
    """
    if len(cells) < min_side_cells * 2:
        return [cells]

    min_row, min_col, max_row, max_col = _bounds(cells)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if width < 16 or height < 10:
        return [cells]

    counts = {col: 0 for col in range(min_col, max_col + 1)}
    for _, col in cells:
        counts[col] += 1
    valley_limit = max(3, int(np.ceil(height * 0.30)))
    edge_guard = 4
    valleys: list[tuple[int, int]] = []
    start: int | None = None
    for col in range(min_col, max_col + 2):
        low = col <= max_col and counts[col] <= valley_limit
        if low and start is None:
            start = col
        elif not low and start is not None:
            end = col - 1
            if (
                end - start + 1 >= 2
                and start > min_col + edge_guard
                and end < max_col - edge_guard
            ):
                valleys.append((start, end))
            start = None
    if not valleys:
        return [cells]

    # Prefer the broadest empty corridor.  It is much less likely to be an
    # internal detail of a character than a one-column dip.
    start, end = max(valleys, key=lambda band: (band[1] - band[0], -band[0]))
    cut = (start + end) // 2
    left = [cell for cell in cells if cell[1] <= cut]
    right = [cell for cell in cells if cell[1] > cut]
    if len(left) < min_side_cells or len(right) < min_side_cells:
        return [cells]
    return (
        _split_bridge_connected_component(left, min_side_cells)
        + _split_bridge_connected_component(right, min_side_cells)
    )


def _split_bridge_connected_components(
    components: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    return [
        piece
        for cells in components
        for piece in _split_bridge_connected_component(cells)
    ]


def _boxes_touch(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    margin: int = 2,
) -> bool:
    """Whether two component boxes belong to the same visual region."""
    a_top, a_left, a_bottom, a_right = first
    b_top, b_left, b_bottom, b_right = second
    return not (
        a_right + margin < b_left
        or b_right + margin < a_left
        or a_bottom + margin < b_top
        or b_bottom + margin < a_top
    )


def _group_adjacent_stroke_groups(
    groups: list[tuple[str, list[tuple[int, int]]]],
) -> list[list[tuple[str, list[tuple[int, int]]]]]:
    """Keep overlapping label parts and outline pieces in one draw region."""
    regions: list[list[tuple[str, list[tuple[int, int]]]]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for group in groups:
        group_box = _bounds(group[1])
        touching = [index for index, box in enumerate(boxes) if _boxes_touch(group_box, box)]
        if not touching:
            regions.append([group])
            boxes.append(group_box)
            continue
        target = touching[0]
        regions[target].append(group)
        top, left, bottom, right = boxes[target]
        boxes[target] = (
            min(top, group_box[0]), min(left, group_box[1]),
            max(bottom, group_box[2]), max(right, group_box[3]),
        )
        # Merge any regions newly bridged by the expanded box.
        for index in reversed(touching[1:]):
            regions[target].extend(regions.pop(index))
            other = boxes.pop(index)
            top, left, bottom, right = boxes[target]
            boxes[target] = (
                min(top, other[0]), min(left, other[1]),
                max(bottom, other[2]), max(right, other[3]),
            )
    return regions


def classify_stroke_groups(
    active: np.ndarray,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Classify connected ink regions as a main subject, text, or local contour."""
    labels, count = _label_components(active)
    components = [
        _component_cells(labels, label)
        for label in range(1, count + 1)
    ]
    components = [cells for cells in components if cells]
    if not components:
        return []

    # A long ground line may connect a mountain, a character, and a crowd.
    # Split that weak connection before any region ordering is decided.
    components = _split_bridge_connected_components(components)

    # 合并小碎片到最近的大区域，避免碎片打断大块文字的连续绘制
    total_cells = sum(len(c) for c in components)
    merge_threshold = max(3, int(total_cells * 0.005))
    components = _merge_small_components(components, merge_threshold)

    subject_index = max(range(len(components)), key=lambda index: len(components[index]))
    groups: list[tuple[str, list[tuple[int, int]], tuple[int, int, int]]] = []
    for index, cells in enumerate(components):
        min_row, min_col, max_row, max_col = _bounds(cells)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        density = len(cells) / (height * width)
        if index == subject_index:
            kind, rank = "subject", 0
        elif height >= 2 and width / height >= 2.2 and density >= 0.5:
            kind, rank = "text", 1
        else:
            kind, rank = "contour", 2
        groups.append((kind, cells, (rank, min_row, min_col)))

    groups.sort(key=lambda group: group[2])
    return [(kind, cells) for kind, cells, _ in groups]


def _density_seed(cells: Sequence[tuple[int, int]], radius: int = 2) -> tuple[int, int]:
    """挑局部邻居最密的格子作为起笔点，模拟“从墨最浓处下笔”。"""
    cell_set = set(cells)
    best = cells[0]
    best_score = -1
    for (r, c) in cells:
        score = sum(
            1
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (r + dr, c + dc) in cell_set
        )
        if score > best_score:
            best_score = score
            best = (r, c)
    return best


def _gradient_walk(cells: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    密度梯度引导的贪心游走：从密度最高的种子格出发，每步选择
    “未访问邻居中局部密度最高、且与来向夹角最小”的格子，
    形成“尽量沿着墨迹、少折返”的连续笔迹。
    无邻居可达时跳到全局最近的未访问格继续。
    """
    if not cells:
        return []

    cell_set = set(cells)
    seed = _density_seed(cells)
    visited: set[tuple[int, int]] = {seed}
    path: list[tuple[int, int]] = [seed]
    current = seed
    prev_dir = (0, 0)

    while len(visited) < len(cells):
        neighbors = [
            (r, c)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc)
            and (r := current[0] + dr, c := current[1] + dc) in cell_set
            and (r, c) not in visited
        ]
        if neighbors:
            def cost(cell: tuple[int, int]) -> tuple:
                # 邻居越多越好（负号取最小）、方向变化越小越好、最后按位置稳定排序
                local = sum(
                    1
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (cell[0] + dr, cell[1] + dc) in cell_set
                    and (cell[0] + dr, cell[1] + dc) not in visited
                )
                step = (cell[0] - current[0], cell[1] - current[1])
                turn = (step[0] - prev_dir[0]) ** 2 + (step[1] - prev_dir[1]) ** 2
                return (-local, turn, cell[0], cell[1])

            nxt = min(neighbors, key=cost)
        else:
            # 断笔：跳到最近的未访问格
            unvisited = [cell for cell in cells if cell not in visited]
            nxt = min(
                unvisited,
                key=lambda cell: (
                    (cell[0] - current[0]) ** 2 + (cell[1] - current[1]) ** 2,
                    cell[0],
                    cell[1],
                ),
            )

        prev_dir = (nxt[0] - current[0], nxt[1] - current[1])
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path


def _nearest_neighbor_order(
    cells: Sequence[tuple[int, int]], seed: tuple[int, int]
) -> list[tuple[int, int]]:
    """从 seed 出发，每步走最近的未访问格，形成连续笔迹。"""
    if not cells:
        return []
    remaining = list(cells)
    ordered: list[tuple[int, int]] = []
    current = seed if seed in remaining else remaining[0]
    while remaining:
        ordered.append(current)
        remaining.remove(current)
        if not remaining:
            break
        current = min(
            remaining,
            key=lambda cell: (cell[0] - ordered[-1][0]) ** 2
            + (cell[1] - ordered[-1][1]) ** 2,
        )
    return ordered


def _text_scan_order(
    cells: Sequence[tuple[int, int]], segment_cols: int = 4
) -> list[tuple[int, int]]:
    """
    文字区域的专用画法：横向按段扫描，模拟写字。
    把格子按列切成若干段（每段 segment_cols 列宽），段间按列从左到右；
    段内用最近邻沿墨迹连续走（而非栅栏式逐行扫），避免“画过一块没画满、
    跳到下一段又从顶部开始”的回头补笔感。
    """
    if not cells:
        return []
    if segment_cols < 1:
        segment_cols = 1
    left_col = min(col for _, col in cells)
    # 按“起始列 // segment_cols”分桶，桶号小（靠左）的先画
    buckets: dict[int, list[tuple[int, int]]] = {}
    for cell in cells:
        bucket_key = (cell[1] - left_col) // segment_cols
        buckets.setdefault(bucket_key, []).append(cell)

    ordered: list[tuple[int, int]] = []
    prev_tail: tuple[int, int] | None = None
    for key in sorted(buckets):
        seg_cells = buckets[key]
        # 段的起点：尽量靠近上一段出口，减少段间跳笔
        if prev_tail is not None:
            seed = min(
                seg_cells,
                key=lambda cell: (cell[0] - prev_tail[0]) ** 2
                + (cell[1] - prev_tail[1]) ** 2,
            )
        else:
            seed = min(seg_cells, key=lambda cell: (cell[0], cell[1]))
        seg_order = _nearest_neighbor_order(seg_cells, seed)
        ordered.extend(seg_order)
        prev_tail = seg_order[-1]
    return ordered


def _order_stream_by_kind(
    kind: str, cells: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """按区域类型选画法：文字横向按段扫，主体/轮廓走密度游走。"""
    if kind == "text":
        return _text_scan_order(cells)
    return _gradient_walk(cells)


def _chain_region_paths(
    groups: list[tuple[str, list[tuple[int, int]]]],
) -> list[tuple[int, int]]:
    """Finish every component in one visual region before leaving it."""
    paths = [_order_stream_by_kind(kind, cells) for kind, cells in groups]
    remaining = [path for path in paths if path]
    ordered: list[tuple[int, int]] = []
    tail: tuple[int, int] | None = None
    while remaining:
        if tail is None:
            pick_index = 0  # groups retain subject/text/contour priority.
        else:
            pick_index = min(
                range(len(remaining)),
                key=lambda index: min(
                    (remaining[index][0][0] - tail[0]) ** 2
                    + (remaining[index][0][1] - tail[1]) ** 2,
                    (remaining[index][-1][0] - tail[0]) ** 2
                    + (remaining[index][-1][1] - tail[1]) ** 2,
                ),
            )
        path = remaining.pop(pick_index)
        if tail is not None and len(path) > 1:
            head_distance = (path[0][0] - tail[0]) ** 2 + (path[0][1] - tail[1]) ** 2
            end_distance = (path[-1][0] - tail[0]) ** 2 + (path[-1][1] - tail[1]) ** 2
            if end_distance < head_distance:
                path.reverse()
        ordered.extend(path)
        tail = path[-1]
    return ordered


def cluster_ink_streams(active: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    把墨迹格按语义聚成若干条墨流：主体(subject) → 文字(text) → 局部轮廓(contour)，
    每条内部按类型选画法（文字按段扫、其余密度游走）；
    墨流之间按“出口到入口最近邻”动态串联，必要时整条反向，减少跳笔。
    返回的是已串联排序好的多条笔迹流。
    """
    if not active.any():
        return []
    groups = classify_stroke_groups(active)
    # A stream is now a complete visual region, not merely one connected
    # component.  Thus a label's border, its characters, and its arrow cannot
    # be interrupted by a different object that happens to be closer.
    regions = _group_adjacent_stroke_groups(groups)
    streams = [_chain_region_paths(region) for region in regions]
    streams = [s for s in streams if s]
    if not streams:
        return []

    # 串联：主体（第一支）开局，之后每次挑入口离当前出口最近的墨流，
    # 并视情况把该墨流整体反向，使其起点更靠近上一支的出口。
    ordered: list[list[tuple[int, int]]] = []
    remaining = list(streams)
    tail: tuple[int, int] | None = None
    while remaining:
        if tail is None:
            pick_idx = 0  # classify 已把主体排在最前
        else:
            def dist_to_tail(stream: list[tuple[int, int]]) -> int:
                head = stream[0]
                return (head[0] - tail[0]) ** 2 + (head[1] - tail[1]) ** 2
            pick_idx = min(range(len(remaining)), key=lambda i: dist_to_tail(remaining[i]))
        pick = remaining.pop(pick_idx)
        # 视情况反向：若尾离 pick 的终点比离起点更近，则反向
        if tail is not None and len(pick) > 1:
            head = pick[0]
            end = pick[-1]
            d_end = (end[0] - tail[0]) ** 2 + (end[1] - tail[1]) ** 2
            d_head = (head[0] - tail[0]) ** 2 + (head[1] - tail[1]) ** 2
            if d_end < d_head:
                pick = pick[::-1]
        ordered.append(pick)
        tail = pick[-1]
    return ordered


def flatten_streams(streams: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for stream in streams for cell in stream]


