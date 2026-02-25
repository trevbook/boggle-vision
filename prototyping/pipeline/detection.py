"""Stage 1–4: YOLO board detection, mask cleanup, quad fitting, perspective warp."""

import cv2
import numpy as np

from .constants import BOARD_CLASSES, WARP_PAD_PCT, YOLO_CONF, YOLO_IMGSZ


def detect_board(image, model, conf=YOLO_CONF, imgsz=YOLO_IMGSZ):
    """Run YOLO segmentation, return (mask, box, det_conf) or (None, None, 0.0)."""
    for conf_thresh in [conf, 0.15, 0.10]:
        results = model(image, conf=conf_thresh, verbose=False, imgsz=imgsz)
        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            continue
        board_indices = [
            i for i, cls in enumerate(result.boxes.cls) if int(cls) in BOARD_CLASSES
        ]
        if not board_indices:
            continue
        best_idx = max(board_indices, key=lambda i: result.boxes.conf[i].item())
        mask = result.masks.data[best_idx].cpu().numpy()
        h, w = image.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8) * 255
        return mask, result.boxes[best_idx], result.boxes.conf[best_idx].item()
    return None, None, 0.0


def cleanup_mask(mask):
    """Morphological close + open, keep only the largest connected component."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


def fit_quad(mask):
    """Fit a 4-corner polygon to the board mask. Returns (corners, method_str)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no_contours"
    contour = max(contours, key=cv2.contourArea)
    for eps_mult in np.arange(0.02, 0.10, 0.005):
        epsilon = eps_mult * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), f"approxPolyDP(eps={eps_mult:.3f})"
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32), "minAreaRect"


def order_corners(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).squeeze()
    return np.array(
        [
            pts[np.argmin(s)],   # TL: smallest x+y
            pts[np.argmin(d)],   # TR: smallest y-x
            pts[np.argmax(s)],   # BR: largest x+y
            pts[np.argmax(d)],   # BL: largest y-x
        ],
        dtype=np.float32,
    )


def warp_board(image, corners, pad_pct=WARP_PAD_PCT):
    """Warp image to a top-down square view of the board.

    pad_pct: expand each corner outward from centroid so edge tiles aren't clipped.
    Returns (warped_image, side_length_px).
    """
    ordered = order_corners(corners)
    if pad_pct > 0:
        centroid = ordered.mean(axis=0)
        ordered = centroid + (1 + pad_pct) * (ordered - centroid)
    side_lengths = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
    size = int(np.ceil(max(side_lengths)))
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped, size
