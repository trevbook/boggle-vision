"""Stage 5: Find tile centers via distance-transform peaks, infer NxN grid."""

from itertools import combinations

import cv2
import numpy as np
from scipy.ndimage import center_of_mass, label as ndlabel, maximum_filter

from .constants import GRID_INSET_RATIO, WARP_PAD_PCT


def find_tile_centers(warped, debug=False):
    """Find tile center positions from local maxima of the distance transform.

    Strategy:
    1. Blur -> Otsu -> binary mask of tiles vs. frame
    2. Distance transform -> peaks at tile centers
    3. Local-maxima detection -> candidate centroids
    4. Border-margin filter: discard peaks within WARP_PAD_PCT of the image edge
       (these are frame/background artifacts from the perspective-warp padding).

    Returns (centroids_Nx2, debug_info_or_None). Centroids are (x, y).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    blur_sigma = max(5, int(w_img * 0.02))
    ksize = blur_sigma * 6 + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    neighborhood = max(3, int(w_img * 0.10))
    local_max = maximum_filter(dist, size=neighborhood) == dist
    local_max &= dist > dist.max() * 0.15

    labeled, n_peaks = ndlabel(local_max)
    if n_peaks == 0:
        return np.empty((0, 2)), None

    raw_centroids = center_of_mass(dist, labeled, range(1, n_peaks + 1))
    centroids = np.array([(x, y) for y, x in raw_centroids])

    # Border-margin filter
    m = WARP_PAD_PCT
    if len(centroids) > 0:
        in_bounds = (
            (centroids[:, 0] > w_img * m)
            & (centroids[:, 0] < w_img * (1 - m))
            & (centroids[:, 1] > h_img * m)
            & (centroids[:, 1] < h_img * (1 - m))
        )
        centroids = centroids[in_bounds]

    debug_info = (
        {"n_peaks": n_peaks, "dist": dist, "blur_sigma": blur_sigma} if debug else None
    )
    return centroids, debug_info


def _group_by_proximity(values, tolerance):
    order = np.argsort(values)
    sorted_vals = values[order]
    groups, current = [], [order[0]]
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[i - 1] < tolerance:
            current.append(order[i])
        else:
            groups.append(np.array(current))
            current = [order[i]]
    groups.append(np.array(current))
    return groups


def _score_combo(groups, values, axis_idx=None):
    ctrs = np.sort(
        [
            np.mean(values[g, axis_idx] if axis_idx is not None else values[g])
            for g in groups
        ]
    )
    gaps = np.diff(ctrs)
    mean_gap = np.mean(gaps)
    if mean_gap <= 0:
        return -1, ctrs
    cv = np.std(gaps) / mean_gap
    return sum(len(g) for g in groups) * max(0.01, 1 - cv), ctrs


def infer_grid_from_centroids(centroids, image_shape, grid_range=(4, 7)):
    """Cluster centroids into a square NxN grid. Returns (N, row_centers, col_centers, tile_spacing)."""
    if len(centroids) < grid_range[0]:
        return 0, np.array([]), np.array([]), 0

    h_img, w_img = image_shape[:2]
    tol = (min(h_img, w_img) / grid_range[1]) * 0.25
    row_groups = _group_by_proximity(centroids[:, 1], tol)
    row_sizes = np.array([len(g) for g in row_groups])

    for N in range(grid_range[1] - 1, grid_range[0] - 1, -1):
        if len(row_groups) < N:
            continue
        sorted_indices = np.argsort(-row_sizes)
        candidates = [int(i) for i in sorted_indices[: 2 * N] if row_sizes[i] >= 2]
        if len(candidates) < N:
            continue

        best_row_score, best_row_result = -1, None
        for combo in combinations(candidates, N):
            groups = [row_groups[i] for i in combo]
            score, ctrs = _score_combo(groups, centroids, axis_idx=1)
            if score > best_row_score:
                best_row_score, best_row_result = score, (groups, ctrs)
        if best_row_result is None:
            continue

        row_groups_sel, row_ctrs = best_row_result
        valid_xs = centroids[np.concatenate(row_groups_sel), 0]
        col_groups = _group_by_proximity(valid_xs, tol)
        col_sizes = np.array([len(g) for g in col_groups])
        if len(col_groups) < N:
            continue
        sorted_c = np.argsort(-col_sizes)
        c_candidates = [int(i) for i in sorted_c[: 2 * N] if col_sizes[i] >= 2]
        if len(c_candidates) < N:
            continue

        best_col_score, best_col_ctrs = -1, None
        for combo in combinations(c_candidates, N):
            c_groups = [col_groups[i] for i in combo]
            score, ctrs = _score_combo(c_groups, valid_xs)
            if score > best_col_score:
                best_col_score, best_col_ctrs = score, ctrs
        if best_col_ctrs is None:
            continue

        y_sp = np.median(np.diff(row_ctrs)) if len(row_ctrs) > 1 else h_img / N
        x_sp = np.median(np.diff(best_col_ctrs)) if len(best_col_ctrs) > 1 else w_img / N
        return N, row_ctrs, best_col_ctrs, (y_sp + x_sp) / 2

    return 0, np.array([]), np.array([]), 0


def _contour_bbox_in_region(region, rx0, ry0, w_img, h_img, min_area_frac=0.15):
    """Find a tight square bbox around the largest white contour in `region`.

    Returns (x1, y1, x2, y2) in warped-image coordinates (same frame as rx0/ry0).
    Falls back to the full region bbox if no dominant contour is found.
    """
    fallback = (rx0, ry0, min(w_img, rx0 + region.shape[1]), min(h_img, ry0 + region.shape[0]))
    if region.size == 0:
        return fallback

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_frac * region.shape[0] * region.shape[1]:
        return fallback

    bx, by, bw, bh = cv2.boundingRect(largest)
    # Make square (max side + small margin) centered on the contour bbox center
    side = max(bw, bh) + 4
    cx_b = rx0 + bx + bw // 2
    cy_b = ry0 + by + bh // 2
    half = side // 2
    return (
        max(0, cx_b - half),
        max(0, cy_b - half),
        min(w_img, cx_b + half),
        min(h_img, cy_b + half),
    )


def extract_tiles_from_grid(
    warped, row_centers, col_centers, tile_size,
    centroids=None, return_bboxes=False, inset_ratio=GRID_INSET_RATIO,
):
    """Crop NxN tile images from warped board using grid centers."""
    h_img, w_img = warped.shape[:2]
    coarse_half = int(tile_size * 0.55)
    snap_tol = tile_size * 0.5
    tiles, bboxes = [], []
    for ry in np.sort(row_centers):
        for cx in np.sort(col_centers):
            cy_f, cx_f = float(ry), float(cx)
            # Step 1: centroid-snap
            if centroids is not None and len(centroids) > 0:
                dists = np.hypot(centroids[:, 0] - cx_f, centroids[:, 1] - cy_f)
                idx = np.argmin(dists)
                if dists[idx] < snap_tol:
                    cx_f, cy_f = centroids[idx, 0], centroids[idx, 1]
            cy_i, cx_i = int(round(cy_f)), int(round(cx_f))
            # Step 2: coarse crop
            rx0 = max(0, cx_i - coarse_half)
            ry0 = max(0, cy_i - coarse_half)
            rx1 = min(w_img, cx_i + coarse_half)
            ry1 = min(h_img, cy_i + coarse_half)
            coarse = warped[ry0:ry1, rx0:rx1]
            # Step 3: contour-snap to tight bbox
            x1, y1, x2, y2 = _contour_bbox_in_region(coarse, rx0, ry0, w_img, h_img)
            tiles.append(warped[y1:y2, x1:x2])
            bboxes.append((x1, y1, x2, y2))
    if return_bboxes:
        return tiles, bboxes
    return tiles
