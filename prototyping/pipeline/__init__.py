"""Boggle Vision CV pipeline — shared module for notebooks and future TS port.

Usage in notebooks (CWD = prototyping/ when launched from that dir):
    import sys
    sys.path.insert(0, str(Path.cwd()))   # adds prototyping/ to path
    from pipeline import analyze_board, BoggleCNN, CLASS_LABELS
"""

from .analyze import analyze_board
from .constants import (
    BOARD_CLASSES,
    CLASS_LABELS,
    GRID_INSET_RATIO,
    TARGET_TILE_SIZE,
    WARP_PAD_PCT,
    YOLO_CONF,
    YOLO_IMGSZ,
)
from .detection import cleanup_mask, detect_board, fit_quad, order_corners, warp_board
from .grid import (
    extract_tiles_from_grid,
    find_tile_centers,
    infer_grid_from_centroids,
)
from .model import BoggleCNN, predict_tiles_batch
from .tiles import correct_tile_perspective, preprocess_tile_v0

__all__ = [
    # Top-level
    "analyze_board",
    # Constants
    "CLASS_LABELS",
    "BOARD_CLASSES",
    "YOLO_CONF",
    "YOLO_IMGSZ",
    "TARGET_TILE_SIZE",
    "GRID_INSET_RATIO",
    "WARP_PAD_PCT",
    # Detection
    "detect_board",
    "cleanup_mask",
    "fit_quad",
    "order_corners",
    "warp_board",
    # Grid
    "find_tile_centers",
    "infer_grid_from_centroids",
    "extract_tiles_from_grid",
    # Tiles
    "correct_tile_perspective",
    "preprocess_tile_v0",
    # Model
    "BoggleCNN",
    "predict_tiles_batch",
]
