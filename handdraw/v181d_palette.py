import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import label


def _palette_regions(image, region_ids, *, visible_mask=None):
    ids = np.asarray(region_ids, np.int32).copy()
    if not np.any(ids > 0):
        ids[:] = 1
    elif np.any(ids == 0):
        nearest = ndimage.distance_transform_edt(ids == 0, return_distances=False,
                                                return_indices=True)
        ids = ids[nearest[0], nearest[1]]
    if visible_mask is not None:
        ids[~visible_mask] = 0
    count = int(ids.max()) + 1
    areas = np.bincount(ids.ravel(), minlength=count)
    active = np.flatnonzero(areas)
    active = active[active > 0]
    means = np.column_stack([
        np.bincount(ids.ravel(), weights=image[:, :, channel].ravel(), minlength=count)[active]
        / areas[active] for channel in range(3)])
    lab = cv2.cvtColor(np.rint(means).astype(np.uint8)[None], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    colors = min(8, len(np.unique(lab, axis=0)))
    if colors == 1:
        assignments = np.zeros(len(active), np.int32)
    else:
        # Weight the palette by image area, not the number of fragmented regions.
        repeats = np.maximum(1, np.rint(areas[active] / areas[active].sum() * 2048).astype(int))
        samples = np.repeat(lab, repeats, axis=0)
        _, _, centers = cv2.kmeans(samples, colors, None,
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .5),
                                  3, cv2.KMEANS_PP_CENTERS)
        assignments = np.argmin(((lab[:, None] - centers[None]) ** 2).sum(axis=2), axis=1)
    lookup = np.zeros(count, np.int32)
    lookup[active] = assignments + 1
    palette = lookup[ids]
    patches = label(palette, connectivity=2).astype(np.int32)
    return palette, patches
