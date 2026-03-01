"""Stage 6-7: Per-tile perspective correction and v0 preprocessing."""

import cv2
import numpy as np

from .constants import TARGET_TILE_SIZE


def correct_tile_perspective(tile_bgr):
    """Straighten a tile crop using its content's oriented bounding box.

    The board-level warp may leave residual per-tile rotation. This finds the
    white tile face via Otsu, computes its min-area rectangle, and warps
    to axis-align it.
    """
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    blur_ksize = max(3, int(min(h, w) * 0.1) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return tile_bgr

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.3 * h * w:
        return tile_bgr

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    s, d = box.sum(axis=1), np.diff(box, axis=1).squeeze()
    ordered = np.array(
        [box[np.argmin(s)], box[np.argmin(d)], box[np.argmax(s)], box[np.argmax(d)]],
        dtype=np.float32,
    )
    side_lengths = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
    size = int(np.ceil(max(side_lengths)))
    if size < 10:
        return tile_bgr

    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    return cv2.warpPerspective(tile_bgr, cv2.getPerspectiveTransform(ordered, dst), (size, size))


def _contour_depth(hierarchy, idx):
    """Walk the parent chain to compute contour nesting depth."""
    depth = 0
    while hierarchy[idx][3] != -1:
        idx = hierarchy[idx][3]
        depth += 1
    return depth


def preprocess_tile_v0(tile_bgr, target_size=TARGET_TILE_SIZE):
    """Replicate v0 preprocessing: adaptive threshold -> contour mask -> center -> resize.

    Produces a target_size x target_size uint8 image (white letter on black background).
    This matches the preprocessing the BoggleCNN was trained on.
    """
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)

    # Trim outer 5% to remove frame-edge artifacts on tiles near the board border
    h_t, w_t = gray.shape[:2]
    m = max(2, int(min(h_t, w_t) * 0.05))
    gray = gray[m : h_t - m, m : w_t - m]

    tile_area = gray.shape[0] * gray.shape[1]
    block_size = max(3, int(tile_area * 0.015) | 1)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 5
    )
    _, thresh = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)

    h = hierarchy[0]
    min_area = tile_area * 0.003
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for i, cnt in enumerate(contours):
        depth = _contour_depth(h, i)
        if depth == 1:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        elif depth >= 2:
            cv2.drawContours(mask, [cnt], -1, 0, cv2.FILLED)

    y_coords, x_coords = np.where(mask > 1)
    if len(x_coords) > 0:
        x_min, y_min = np.min(x_coords), np.min(y_coords)
        x_max, y_max = np.max(x_coords), np.max(y_coords)
        cropped = mask[y_min:y_max, x_min:x_max]
        img_h, img_w = mask.shape
        centered = np.zeros_like(mask)
        start_x = (img_w - cropped.shape[1]) // 2
        start_y = (img_h - cropped.shape[0]) // 2
        centered[start_y : start_y + cropped.shape[0], start_x : start_x + cropped.shape[1]] = cropped
        mask = centered

    return cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_AREA)
