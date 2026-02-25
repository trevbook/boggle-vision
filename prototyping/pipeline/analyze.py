"""End-to-end pipeline: image path → letter matrix."""

import cv2
import numpy as np

from .detection import cleanup_mask, detect_board, fit_quad, warp_board
from .grid import extract_tiles_from_grid, find_tile_centers, infer_grid_from_centroids
from .model import predict_tiles_batch
from .tiles import correct_tile_perspective, preprocess_tile_v0


def analyze_board(image_path, yolo_model, cnn_model):
    """Full CV pipeline: image path → predicted letter sequence.

    Args:
        image_path: str or Path to the board photo.
        yolo_model: loaded Ultralytics YOLO model.
        cnn_model: loaded BoggleCNN in eval mode.

    Returns a dict with keys:
        letters       list[str]    — flat, row-major (length = grid_size²)
        confidences   list[float]  — parallel to letters
        grid_size     int
        mean_confidence float
        min_confidence  float
        warped        np.ndarray   — perspective-corrected board image
        tiles         list[np.ndarray] — individual tile crops
        det_conf      float        — YOLO detection confidence
        quad_method   str
        warp_size     int          — side length of the warped image (px)

    On failure, returns {"error": <message>} with no other keys guaranteed.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": f"Failed to load {image_path}"}

    # Stage 1–2: detect board + clean mask
    mask, box, det_conf = detect_board(image, yolo_model)
    if mask is None:
        return {"error": "No board detected"}
    clean = cleanup_mask(mask)

    # Stage 3–4: quad fit + perspective warp
    corners, quad_method = fit_quad(clean)
    if corners is None:
        return {"error": "Quad fitting failed"}
    warped_img, warp_sz = warp_board(image, corners)

    # Stage 5: tile center detection + grid inference
    tile_centroids, _ = find_tile_centers(warped_img)
    if len(tile_centroids) < 4:
        return {"error": f"Only {len(tile_centroids)} tile peaks found"}
    gs, rows, cols, tsize = infer_grid_from_centroids(tile_centroids, warped_img.shape)
    if gs == 0:
        return {"error": "No valid grid inferred"}

    # Stage 6: extract + perspective-correct individual tiles
    tile_imgs = [
        correct_tile_perspective(t)
        for t in extract_tiles_from_grid(warped_img, rows, cols, tsize, centroids=tile_centroids)
    ]

    # Stage 7–8: preprocess + CNN inference
    preprocessed = [preprocess_tile_v0(t) for t in tile_imgs]
    letters, confs = predict_tiles_batch(cnn_model, preprocessed)

    return {
        "letters": letters,
        "confidences": confs,
        "grid_size": gs,
        "mean_confidence": float(np.mean(confs)),
        "min_confidence": float(np.min(confs)),
        "warped": warped_img,
        "tiles": tile_imgs,
        "det_conf": det_conf,
        "quad_method": quad_method,
        "warp_size": warp_sz,
    }
