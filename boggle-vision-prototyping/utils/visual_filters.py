# General import statements
import pandas as pd
from tqdm import tqdm
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np


def transform_and_place_heatmap(cropped_image, contour, heatmap):
    """
    This function takes a single contour and its corresponding heatmap,
    transforms the heatmap to align with the contour, and places it on
    the cropped_image.

    Args:
    - cropped_image: The original image of the board.
    - contour: The coordinates of the edge contour of a tile.
    - heatmap: The activation heatmap for the corresponding tile.

    Returns:
    - transformed_image: The image with the heatmap transformed and placed on it.
    """

    # Get the minimum area rectangle for the contour
    contour = np.array(contour, dtype=np.float32)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Get width and height of the bounding box
    width, height = rect[1][0], rect[1][1]

    # Correct the orientation of the box if necessary
    if rect[2] < -45:
        width, height = height, width
    width, height = int(width), int(height)

    # Invert the heatmap values
    # heatmap = 255 - heatmap

    # Resize heatmap to match the size of the bounding box
    resized_heatmap = cv2.resize(
        heatmap, (width, height), interpolation=cv2.INTER_CUBIC
    )

    # Points of the rectangle on the original image
    pts1 = np.float32(box)

    # Points of the rectangle on the heatmap
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Compute the transformation matrix
    M = cv2.getPerspectiveTransform(pts2, pts1)

    # Apply the transformation matrix to the heatmap
    dst = cv2.warpPerspective(
        resized_heatmap, M, (cropped_image.shape[1], cropped_image.shape[0])
    )

    # Create a mask to combine with the original image
    mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, box, 255)

    # If the cropped image is color (3 channels), convert the mask and dst to 3-channel
    if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
        mask = np.stack([mask] * 3, axis=-1)
        dst = np.stack([dst] * 3, axis=-1)

    # Combine the mask and the heatmap
    combined = cv2.bitwise_and(dst, mask)

    # Add the combined image to the original cropped_image
    transformed_image = cv2.addWeighted(cropped_image, 1, combined, 1, 0)

    return transformed_image


def generate_activation_heatmap_filter(
    cropped_image, activation_viz_list, tile_idx_to_contour_dict
):
    """
    This method will create an "activation heatmap" filter.
    """

    # Make a blank image the same size as cropped_image
    activation_heatmap_img = np.zeros_like(cropped_image)
    for tile_idx in range(len(activation_viz_list)):
        activation_heatmap_img = transform_and_place_heatmap(
            activation_heatmap_img,
            contour=tile_idx_to_contour_dict[tile_idx],
            heatmap=activation_viz_list[tile_idx],
        )

    # Create a greyscale image
    greyscale_img = cv2.cvtColor(activation_heatmap_img, cv2.COLOR_BGR2GRAY)

    # Return the greyscale image
    return greyscale_img


def apply_canny_edge_detection(
    img,
    resize_height=750,
    return_original=False,
    thicken_edges=True,
    edge_thickness_kernel_size=3,
):
    """
    Apply Canny Edge Detection to an image, with options to resize, return original, and thicken edges.

    :param img: Input image as a NumPy array.
    :param resize_height: Height to resize the image to, maintaining aspect ratio.
    :param return_original: Whether to return the original image along with the edges.
    :param thicken_edges: Whether to thicken the edges in the output.
    :return: Edges, or edges with original image depending on 'return_original'.
    """

    # Resize the image if necessary
    if resize_height is not None and img.shape[0] >= resize_height:
        scale = resize_height / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # Convert to grayscale and apply Gaussian blur
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    # Thicken the edges if requested
    if thicken_edges:
        kernel = np.ones(
            (edge_thickness_kernel_size, edge_thickness_kernel_size), np.uint8
        )  # You can adjust the kernel size for different thickness
        edges = cv2.dilate(edges, kernel, iterations=1)

    # Return the resulting image
    if return_original:
        return edges, img
    else:
        return edges
