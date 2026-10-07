"""Native V8 grid ink order."""
import threading
import cv2
import numpy as np
from . import grid_stream
_KMEANS_LOCK = threading.Lock()

def _grid_ink(image, visible_mask=None, *, ink_map=None, edge=10, block_size=15, threshold_c=10):
    h, w = image.shape[:2]
    if ink_map is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, threshold_c)
    else:
        if (not isinstance(ink_map, np.ndarray) or ink_map.shape != (h, w)
                or ink_map.dtype != np.uint8 or np.any((ink_map != 0) & (ink_map != 255))):
            raise ValueError("ink_map must be a matching uint8 0/255 mask (0 = ink).")
        threshold = ink_map.copy()
    if visible_mask is not None:
        threshold[~visible_mask] = 255
    ink = threshold == 0
    padded = cv2.copyMakeBorder(threshold, 0, (-h)%edge, 0, (-w)%edge, cv2.BORDER_CONSTANT, value=255)
    backend = grid_stream
    cells = backend.flatten_streams(backend.cluster_ink_streams(backend._active_mask(padded, edge, 10)))
    structure = np.full((h, w), 255, np.uint8)
    ink_path = []
    for index, (row, col) in enumerate(cells):
        bounds = np.s_[row*edge:min(h,(row+1)*edge), col*edge:min(w,(col+1)*edge)]
        structure[bounds][ink[bounds]] = round(index/max(1,len(cells)-1)*254)
        ink_path.append([index/max(1,len(cells)-1), min(w-1,col*edge+edge//2), min(h-1,row*edge+edge//2)])
    return ink, structure, ink_path


