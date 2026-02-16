# This file contains various methods that are used
# to detect the board in the image and return the
# letter sequence of the board.

# =================================================
#                      SETUP
# =================================================
# The code below will set up the rest of this file.

# Import statements
import cv2
import pandas as pd
from pathlib import Path
import math
from matplotlib import pyplot as plt
import numpy as np
from statistics import mode
import utils
import cv2

# import pytesseract
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import utils.settings as settings

# Torch-specific import statements
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.io import read_image

# Custom-built modules and settings
from utils.settings import allowed_boggle_tiles
from utils.cnn import SaveFeatures
from utils.visual_filters import apply_canny_edge_detection


# =================================================
#                      METHODS
# =================================================
# Below, you'll find a number of methods that are
# used for board detection.


def convert_to_greyscale(img):
    """
    This method will convert the image to greyscale using
    cv2's cvtColor method.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_binary_thresholding(img, threshold=127):
    """
    This method will apply a binary thresholding to an image
    using the cv2.threshold method. The `threshold` parameter
    will be used to determine the threshold value.
    """
    ret, thresholded_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresholded_image


def detect_contours(
    img,
    hierarchy_algorithm=cv2.RETR_TREE,
    contour_approximation=cv2.CHAIN_APPROX_NONE,
    apply_preprocessing=False,
):
    """
    This method will apply a contour detection algorithm to the given image.
    The return value will be a list of contours, and a list of hierarchy values.
    """

    # If `apply_preprocessing` is True, we'll apply some preprocessing to the image
    if apply_preprocessing:
        # Convert the image to greyscale
        img = convert_to_greyscale(img)

        # Apply a binary thresholding to the image
        img = apply_binary_thresholding(img)

    contours, hierarchy = cv2.findContours(
        img, hierarchy_algorithm, contour_approximation
    )
    return contours, hierarchy


def show_cv2_image(img):
    """
    This method will display the given image using matplotlib's pyplot.
    Since cv2 uses BGR, and matplotlib uses RGB, we need to convert the image
    to RGB before displaying it.
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def hierarchy_to_dataframe(hierarchy):
    """
    This function will convert the hierarchy produced by cv2.findContours into a pandas dataframe.
    The DataFrame will have information about the contour index, hierarchy level, parent contour index,
    and children contour indices.
    """

    # TODO: I need to refine this method a little more, and make sure that it works properly.

    def dfs(contour_idx, level, max_levels=4):
        """
        This is a helper method that performs a depth-first search on the contour hierarchy.
        """

        # If the current level is at or beyond the maximum, return immediately.
        if max_levels is not None and level >= max_levels:
            return

        _, _, first_child, parent = hierarchy[0, contour_idx]
        children = []
        if first_child != -1:
            child = first_child
            while True:
                children.append(child)
                next_sibling, _, _, _ = hierarchy[0, child]
                if next_sibling != -1:
                    child = next_sibling
                else:
                    break

        data.append(
            [
                contour_idx,
                level,
                parent if parent != -1 else None,
                children if children else None,
            ]
        )
        for child in children:
            dfs(child, level + 1, max_levels)

    data = []
    root_contours = np.where(hierarchy[0, :, 3] == -1)[0]

    # Run DFS from each root contour
    for contour_idx in root_contours:
        dfs(contour_idx, 0)

    # Create the DataFrame containing the contour hierarchy
    df = pd.DataFrame(
        data, columns=["contour_idx", "hierarchy_level", "parent", "children"]
    )

    # Add a n_children column
    df["n_children"] = df["children"].apply(lambda x: len(x) if x else 0)

    return df.drop_duplicates(subset=["contour_idx"])


def approximate_polygon_from_contour(contour, epsilon=0.05):
    """
    This method will use cv2's approxPolyDP method to approximate
    a polygon from the given contour.
    """

    # Approximate the polygon from the contour
    perimeter = cv2.arcLength(contour, True)
    approximated_polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
    return approximated_polygon


def expand_contour(input_image, contour, dilation_size=5):
    """
    This method will expand the given contour by dilating it. The dilation
    will be performed using a kernel of size `dilation_size`.
    """

    # Create an empty mask to draw the contour onto
    mask = np.zeros(input_image.shape[:2], dtype=np.uint8)

    # Draw the contour onto the mask (using a thickness of -1 to fill the contour)
    cv2.drawContours(mask, [contour], -1, (255), thickness=-1)

    # Create a structuring element for the dilation
    kernel = np.ones((dilation_size, dilation_size), np.uint8)

    # Dilate the mask
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Find the contours of the dilated mask
    dilated_contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # The dilated contour will be the one with the largest area
    dilated_contour = max(dilated_contours, key=cv2.contourArea)

    # Simplify the dilated contour
    dilated_contour = cv2.approxPolyDP(
        dilated_contour,
        0.02 * cv2.arcLength(dilated_contour, True),
        True,
    )

    return dilated_contour


def detect_boggle_board_contour(
    input_img,
    n_top_contours_to_consider=20,
    min_board_area_threshold=0.2,
    max_board_area_threshold=0.9,
    board_contour_expansion_size=5,
    polygon_approximation_epsilon=0.05,
    binary_threshold_value=127,
):
    """
    This method will try to detect the contour of the Boggle board in
    the image located at `input_image_path`. If there's a board
    found, then the method qill return the contour of the board.
    """

    # Determine the entire area of the input image
    img_area = input_img.shape[0] * input_img.shape[1]

    # Convert the image to greyscale
    greyscale_img = convert_to_greyscale(input_img)

    # Apply some Gaussian blurring to the image
    # greyscale_img = cv2.GaussianBlur(greyscale_img, (5, 5), 0)

    # Apply a binary thresholding to the image
    thresholded_img = apply_binary_thresholding(
        greyscale_img, threshold=binary_threshold_value
    )

    # Detect the contours in the image
    contours, hierarchy = detect_contours(thresholded_img)

    # Retreive the hierarchy DataFrame
    hierarchy_df = hierarchy_to_dataframe(hierarchy)

    # Add the contours to the image
    hierarchy_df["contour"] = hierarchy_df["contour_idx"].apply(
        lambda idx: contours[idx] if idx >= 0 else None
    )

    # Determine the contours to consider
    contours_to_consider = hierarchy_df.sort_values("n_children", ascending=False).head(
        n_top_contours_to_consider
    )

    # Approximate the contours to polygons
    contours_to_consider["approx_polygon"] = contours_to_consider["contour"].apply(
        lambda contour: approximate_polygon_from_contour(
            contour, epsilon=polygon_approximation_epsilon
        )
    )

    # Determine the number of sides in each approximated contour
    contours_to_consider["approx_polygon_n_sides"] = contours_to_consider[
        "approx_polygon"
    ].apply(lambda x: len(x))

    # Subset the contours to only include contours with 4 sides
    contours_to_consider = contours_to_consider.query(
        "approx_polygon_n_sides == 4"
    ).copy()

    # If the contours_to_consider is empty, then we need to raise an error
    if len(contours_to_consider) == 0:
        raise Exception("There were no square contours detected in the image.")

    # Determine the area of each approximated polygon
    contours_to_consider["approx_polygon_area"] = contours_to_consider[
        "approx_polygon"
    ].apply(lambda x: cv2.contourArea(x))

    # Determine how much of the image is covered by the contour
    contours_to_consider["approx_polygon_area_pct_of_image"] = (
        contours_to_consider["approx_polygon_area"] / img_area
    )

    # Determine the contour with the largest area within a certain threshold
    contours_to_consider = (
        contours_to_consider.query(
            "approx_polygon_area_pct_of_image >= @min_board_area_threshold"
        )
        .query("approx_polygon_area_pct_of_image <= @max_board_area_threshold")
        .copy()
    )
    if len(contours_to_consider) == 0:
        raise Exception(
            "No contours were found that were within the specified area thresholds."
        )
    largest_contour_row = contours_to_consider.sort_values(
        "approx_polygon_area_pct_of_image", ascending=False
    ).iloc[0]

    # Convert the DataFrame row to a dictionary
    largest_contour_dict = largest_contour_row.to_dict()

    # Expand the contour a bit
    largest_contour_dict["expanded_contour"] = expand_contour(
        input_img,
        largest_contour_dict["contour"],
        dilation_size=board_contour_expansion_size,
    )

    # Finally, we're going to approximate the expanded contour to a polygon,
    # and then return this polygon.
    return approximate_polygon_from_contour(
        largest_contour_dict["expanded_contour"], epsilon=polygon_approximation_epsilon
    )


def draw_contours(img, contours, color=(0, 255, 0), thickness=3, return_img=False):
    """
    This method will show the contours on an image. If the
    `return_img` parameter is set to True, then the method
    will return the image with the contours drawn on it.
    If the `return_img` parameter is set to False, then the
    method will display the image using matplotlib's pyplot.
    """

    # Copy the image so that we don't modify the original image
    img_copy = img.copy()

    # Draw the contours on the image
    img_copy = cv2.drawContours(
        img_copy,
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
    )

    # If the return_img parameter is set to True, then we'll return the image
    if return_img:
        return img_copy

    # Otherwise, we'll display the image using pyplot
    plt.imshow(img_copy)
    plt.show()


def order_points(pts):
    """
    This function takes a list of four points and returns them in ordered manner
    (top-left, top-right, bottom-right, bottom-left)
    """
    # Sort the points according to their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # Get the left-most and right-most points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # Within the left-most points, sort them by their y-coordinate so we get the top-left and bottom-left points
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # Now, for the right-most points, we already know that our top-right point will have the smaller y-value
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most

    # Return the ordered coordinates
    return np.array([tl, tr, br, bl], dtype="float32")


def center_letter_image(img):
    """
    This method will take a grayscale image with a white object (letter)
    on a black background and return a new image where the white object is centered.
    """

    # Find all white (also shades of whites)
    # pixels and get their coordinates
    img_height, img_width = img.shape
    y_coords, x_coords = np.where(img > 1)

    # If the image is all black, just return the original image
    if len(x_coords) == 0 or len(y_coords) == 0:
        return img

    # Get the bounding rectangle
    x_min, y_min = np.min(x_coords), np.min(y_coords)
    x_max, y_max = np.max(x_coords), np.max(y_coords)

    # Crop the image to the bounding rectangle
    cropped_img = img[y_min:y_max, x_min:x_max]

    # Get the size of the cropped image
    cropped_img_height, cropped_img_width = cropped_img.shape

    # Create a new image with the same size as the original image
    centered_img = np.zeros_like(img)

    # Get the starting coordinates to center the cropped image
    start_x = (img_width - cropped_img_width) // 2
    start_y = (img_height - cropped_img_height) // 2

    # Paste the cropped image into the center of the new image
    centered_img[
        start_y : start_y + cropped_img_height, start_x : start_x + cropped_img_width
    ] = cropped_img

    return centered_img


def warp_perspective_to_top_down(img, contour):
    """
    This method will take an image and a contour and warp the image to a top-down view.
    The method will return the warped image.
    """

    # Determine the corner points of the board
    corner_points = [contour_corner[0].tolist() for contour_corner in contour]

    # Order the points accordingly
    corner_points = order_points(np.array(corner_points))

    # Unpack the corner points
    pt_A, pt_B, pt_C, pt_D = corner_points

    # Use the L2 norm to calculate the width and height of the new image
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Determine whether the max width or max height is larger
    largest = max(maxWidth, maxHeight)
    maxWidth = largest
    maxHeight = largest

    # Compute the perspective transform matrix
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32(
        [[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]]
    )
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    warped_image = cv2.warpPerspective(
        img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR
    )

    # Rotate the image 90 degrees clockwise
    warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)

    # Flip the image horizontally and return it
    return cv2.flip(warped_image, 1)


def resize_image(image, desired_height):
    """
    This function resizes an image to a desired height while maintaining the aspect ratio.
    """

    # Resize image
    height, width = image.shape[:2]
    aspect_ratio = width / height
    desired_width = int(desired_height * aspect_ratio)
    return cv2.resize(
        image, (desired_width, desired_height), interpolation=cv2.INTER_AREA
    )


def minimum_area_rectangle_from_contour(contour):
    """
    This method will return the minimum area rectangle from a contour.
    The rectangle will be in the form of a contour.
    """

    # Find the minimum area rectangle
    min_area_rect = cv2.minAreaRect(contour)

    # Calculate the box points
    box_points = cv2.boxPoints(min_area_rect)

    # Convert the box points to integers
    box_points = np.int0(box_points)

    # Create a new contour from the box points and return it
    return box_points.reshape((-1, 1, 2))


def detect_tile_contours(
    top_down_board_image,
    binary_threshold_value=100,
    min_tile_area_percentage=0.0003,
    max_tile_area_percentage=0.02,
    tile_size_difference_threshold=0.3,
    polygon_approximation_epsilon=0.02,
):
    """
    This method will detect the contours of the tiles in the top-down board image.
    It will return a DataFrame containing the contours of the tiles, as well as the
    sequence of the tiles.
    """

    # Calculate the area of the image
    input_board_image_area = (
        top_down_board_image.shape[0] * top_down_board_image.shape[1]
    )

    # Convert the image to grayscale
    grayscale_image = convert_to_greyscale(top_down_board_image)

    # Apply some binary thresholding to the image
    binary_image = apply_binary_thresholding(grayscale_image, binary_threshold_value)

    # Detect the contours of the image
    contours, hierarchy = detect_contours(binary_image)

    # Make a dataframe of the contours and their hierarchy
    hierarchy_df = hierarchy_to_dataframe(hierarchy)

    # Add the contours to the image
    hierarchy_df["contour"] = hierarchy_df["contour_idx"].apply(
        lambda idx: contours[idx] if idx >= 0 else None
    )

    # Calculate the area of the contours
    hierarchy_df["contour_area"] = hierarchy_df["contour"].apply(
        lambda contour: cv2.contourArea(contour) if contour is not None else 0
    )
    hierarchy_df["contour_area_pct_of_image"] = (
        hierarchy_df["contour_area"] / input_board_image_area
    )

    # Sort by the total area of the contour
    hierarchy_df = hierarchy_df.sort_values(
        "contour_area_pct_of_image", ascending=False
    )

    # Filter out some of the noise from the contour DataFrame
    contours_to_consider = (
        hierarchy_df.query("contour_area_pct_of_image >= @min_tile_area_percentage")
        .query("contour_area_pct_of_image <= @max_tile_area_percentage")
        .copy()
    )

    # If there are no tiles that meet the minimum area threshold, then raise an exception
    if len(contours_to_consider) == 0:
        raise Exception(f"No tiles found that meet the specified area threshold.")

    # Calculate the difference in percentage of image area between each successively smaller contour
    difference_pct = []
    prev_contour_area = 0
    for idx, row in enumerate(list(contours_to_consider.itertuples())):
        # Extract the area of the current contour
        cur_contour_area = row.contour_area_pct_of_image

        # If we're looking at the largest contour, there is no previous contour to compare to
        if idx == 0:
            prev_contour_area = cur_contour_area
            difference_pct.append(None)
            continue

        # Calculate the difference in percentage of image area between each successively smaller contour
        difference_pct.append(
            (prev_contour_area - cur_contour_area) / prev_contour_area
        )
        prev_contour_area = cur_contour_area

    # Add this difference percentage to the dataframe
    contours_to_consider["difference_pct"] = difference_pct

    # Iterate through each contour, and stop when we find a difference in area between the current contour and the previous contour
    # that's larger than the tile_size_difference_threshold
    break_idx = None
    tile_df_rows = []
    prev_row_area_pct = contours_to_consider.iloc[0].contour_area_pct_of_image
    for idx, row in enumerate(list(contours_to_consider.itertuples())):
        # If we're looking at the first tile, then continue
        if idx == 0:
            continue

        # If the difference in area between the current contour and the previous contour is larger than the tile_size_difference_threshold,
        # then we've reached the end of the board
        if (
            row.difference_pct > tile_size_difference_threshold
            and len(tile_df_rows) > 0
        ):
            break

        # Otherwise, we're going to add the current row to the tile_df_rows list
        elif row.difference_pct <= tile_size_difference_threshold:
            # If there haven't been any tiles added to the tile_df_rows list yet, then add the previous row to the list
            # as well as the current row
            if len(tile_df_rows) == 0:
                tile_df_rows.append(contours_to_consider.iloc[idx - 1].to_dict())

            tile_df_rows.append(row._asdict())
            prev_row_area_pct = row.contour_area_pct_of_image

    # Make a DataFrame out of all of the tile contours we've found.
    # The tile_df_rows list is a list of Series objects, each of which is a row in the DataFrame.
    tile_contours_df = pd.DataFrame(tile_df_rows)

    # Determine the square root of the number of tiles in the board
    # (this will be the number of rows and columns in the board)
    num_rows = math.sqrt(len(tile_contours_df))

    # If the length of the tile_contours_df isn't square, then we've found an invalid board
    if not num_rows % 1 == 0:
        raise Exception(
            f"Invalid board detected. Only found {len(tile_contours_df)} tiles."
        )
    else:
        num_rows = int(num_rows)

    # Calculate minimuim area bounding rectangle for each tile
    tile_contours_df["min_area_rectangle_contour"] = tile_contours_df["contour"].apply(
        lambda contour: minimum_area_rectangle_from_contour(contour)
    )

    # For each minimum area bounding rectangle, calculate the lowest x and y coordinates
    tile_contours_df["min_area_rectangle_lowest_x"] = tile_contours_df[
        "min_area_rectangle_contour"
    ].apply(lambda contour: min([point[0][0] for point in contour]))
    tile_contours_df["min_area_rectangle_lowest_y"] = tile_contours_df[
        "min_area_rectangle_contour"
    ].apply(lambda contour: min([point[0][1] for point in contour]))

    # Make a copy of the tile_contours_df with relevant columns
    tile_contours_and_min_coordinates_df = (
        tile_contours_df[
            [
                "contour_idx",
                "min_area_rectangle_lowest_y",
                "min_area_rectangle_lowest_x",
            ]
        ]
        .sort_values("min_area_rectangle_lowest_y")
        .copy()
    )

    # Iterate through each row and determine the sorting
    # order of the columns
    rearranged_tile_df_by_row_subsets = []
    for cur_row_num in range(int(num_rows)):
        # Subset the tile_contours_and_min_coordinates_df
        cur_tile_subset = tile_contours_and_min_coordinates_df.iloc[
            cur_row_num * num_rows : (cur_row_num + 1) * num_rows
        ]

        # Sort the subset by the x-coordinate of the lowest point
        cur_tile_subset = cur_tile_subset.sort_values("min_area_rectangle_lowest_x")

        # Now, append the subset to the rearranged_tile_df_by_row_subsets
        rearranged_tile_df_by_row_subsets.append(cur_tile_subset)

    # Concatenate the rearranged_tile_df_by_row_subsets
    rearranged_tile_df = pd.concat(rearranged_tile_df_by_row_subsets)

    # Add a column indicating the sequence of tiles
    rearranged_tile_df = (
        rearranged_tile_df.reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "tile_sequence_idx"})
    )[["tile_sequence_idx", "contour_idx"]]

    # Merge this DataFrame back into the tile_contours_df
    tile_contours_df = tile_contours_df.merge(
        rearranged_tile_df, on="contour_idx", how="left"
    )

    # Drop the non-relevant columns and sort by the tile sequence index
    tile_contours_df = (
        tile_contours_df[
            [
                "min_area_rectangle_contour",
                "tile_sequence_idx",
            ]
        ]
        .sort_values("tile_sequence_idx", ascending=True)
        .rename(columns={"min_area_rectangle_contour": "contour"})
    )

    # Return the tile contours DataFrame
    return tile_contours_df


def identify_underline_contours(
    input_img, input_hierarchy_df, min_contour_in_underline=1
):
    """
    This function will take in a modified contour hierarchy DataFrame, and return a list
    of the contours that're suspected to be underlines. This is helpful in detecting
    W's, M's, and Z's, which are underlined in Boggle.
    """
    try:
        # Filter out any contours that're larger than ~3% of the image
        input_hierarchy_df = input_hierarchy_df.query(
            "contour_pct_of_image_area <= 0.03 & contour_pct_of_image_area >= 0.001"
        )

        input_img_size = input_img.shape[0] * input_img.shape[1]
        contour_area_sorted_df = input_hierarchy_df.sort_values(
            "contour_area", ascending=False
        )
        # Add a column indicating the difference in size between each contour
        difference = []
        difference_pct = []
        prev_row_area = contour_area_sorted_df.iloc[0].contour_area * 2
        for idx, row in contour_area_sorted_df.iterrows():
            difference.append(prev_row_area - row.contour_area)
            difference_pct.append((prev_row_area - row.contour_area) / prev_row_area)
            prev_row_area = row.contour_area
        contour_area_sorted_df["difference"] = difference
        contour_area_sorted_df["difference_pct"] = difference_pct
        contour_area_sorted_df = contour_area_sorted_df.reset_index(drop=True)

        # Iterate through each contour and try and locate the tiles
        size_difference_threshold = 0.175
        underline_contour_idxs = []
        prev_row = contour_area_sorted_df.iloc[0]
        for idx, row in enumerate(list(contour_area_sorted_df.itertuples())):
            if idx == 0:
                continue
            if row.difference_pct <= size_difference_threshold:
                underline_contour_idxs.append(prev_row.contour_idx)
                underline_contour_idxs.append(row.contour_idx)
            elif (
                row.difference_pct > size_difference_threshold
                and len(underline_contour_idxs) > 0
            ):
                break
            prev_row = row
        underline_contour_idxs = list(set(underline_contour_idxs))

        # Now, we're going to subset the contour_area_sorted_df to only show the contours that we're interested in
        underline_contour_df = contour_area_sorted_df[
            contour_area_sorted_df.contour_idx.isin(underline_contour_idxs)
        ]

        # If there are less than 3 contours, then we're going to return an empty list
        if len(underline_contour_df) < min_contour_in_underline:
            return []

        # Determine the range of midpoint x and y values
        midpoint_x_range = (
            underline_contour_df["contour_midpoint_x"].max()
            - underline_contour_df["contour_midpoint_x"].min()
        )
        midpoint_y_range = (
            underline_contour_df["contour_midpoint_y"].max()
            - underline_contour_df["contour_midpoint_y"].min()
        )
        midpoint_x_range_percent = midpoint_x_range / input_img_size
        midpoint_y_range_percent = midpoint_y_range / input_img_size

        if midpoint_x_range_percent <= 0.001 or midpoint_y_range_percent <= 0.001:
            return list(underline_contour_df["contour_idx"])
        else:
            return []

    except Exception as e:
        return []


def extract_tile_images(
    top_down_board_image,
    tile_contours_df,
    min_contour_pct_of_total_area=0.03,
    polygon_approximation_epsilon=0.025,
    min_contour_in_underline=2,
    binary_threshold_value=100,
    adaptive_threshold_kernel_size_relative=0.02,
    adaptive_threshold_C=5,
    resize_size=50,
):
    """
    This method will extract images for each of the tiles specified
    by the `tile_contours_df`. The images will be extracted from the
    `top_down_board_image`.

    The images will be cleaned up as much as possible, so that they
    depict the tile as clearly as possible. The returned images will
    be grayscale images with a black background and a white foreground.

    This method will return a dictionary mapping the tile sequence
    index to the extracted image.
    """

    # We're going to store each of the extracted images in this dictionary
    tile_idx_to_extracted_image = {}

    # Determine a solid value for the adaptive threshold kernel size
    mean_tile_area = (
        tile_contours_df["contour"].apply(lambda x: cv2.contourArea(x)).mean()
    )
    optimal_kernel_size = int(mean_tile_area * adaptive_threshold_kernel_size_relative)
    if optimal_kernel_size % 2 == 0:
        optimal_kernel_size += 1

    # We're going to keep a dict of the tile contours that contain
    # special properties
    special_tile_contours = {"rotate_fixed": {}, "is_block": [], "is_i": []}

    # Iterate through each of the tile contours
    for idx, row in enumerate(list(tile_contours_df.itertuples())):
        # First, we're going to warp the image so that we're looking
        # at the tile from a top-down perspective
        top_down_tile_image = warp_perspective_to_top_down(
            img=top_down_board_image, contour=row.contour
        )

        # Calculate the area of the top-down tile image
        top_down_tile_image_area = (
            top_down_tile_image.shape[0] * top_down_tile_image.shape[1]
        )

        # Make this image grayscale
        grayscale_tile_image = convert_to_greyscale(top_down_tile_image)

        # Apply some adaptive thresholding to the image
        thresholded_image = cv2.adaptiveThreshold(
            grayscale_tile_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=optimal_kernel_size,
            C=adaptive_threshold_C,
        )

        # Now, apply some binary thresholding to the image
        thresholded_image = apply_binary_thresholding(
            thresholded_image, threshold=binary_threshold_value
        )

        # Now, we're going to detect the contours from the warped image
        contours, hierarchy = detect_contours(
            thresholded_image, apply_preprocessing=False
        )

        # Make a hierarchy dataframe, and copy it
        hierarchy_df = hierarchy_to_dataframe(hierarchy)
        original_hierarchy_df = hierarchy_df.copy()

        # Add the areaas of the contours to the dataframe
        hierarchy_df["contour_area"] = [
            cv2.contourArea(contour) for contour in contours
        ]
        hierarchy_df["contour_pct_of_image_area"] = (
            hierarchy_df["contour_area"] / top_down_tile_image_area
        )

        # Figure out the midpoint of each contour
        hierarchy_df["contour_midpoint"] = hierarchy_df.apply(
            lambda x: np.mean(contours[x.contour_idx], axis=0)[0], axis=1
        )
        hierarchy_df["contour_midpoint_x"] = hierarchy_df["contour_midpoint"].apply(
            lambda x: x[0]
        )
        hierarchy_df["contour_midpoint_y"] = hierarchy_df["contour_midpoint"].apply(
            lambda x: x[1]
        )
        hierarchy_df = hierarchy_df.drop(columns=["contour_midpoint"])

        # Determine whether or not there are any "outline" contours
        # (these will be found in Z's, M's, W's, etc.)
        underline_contours = identify_underline_contours(
            top_down_tile_image,
            hierarchy_df,
            min_contour_in_underline=min_contour_in_underline,
        )

        # Filter out any level-1 contours that're too small. Make sure to
        # keep the underline contours, though
        hierarchy_df = (
            pd.concat(
                [
                    hierarchy_df.query("hierarchy_level > 1"),
                    hierarchy_df.query(
                        "hierarchy_level == 1 and contour_pct_of_image_area >= @min_contour_pct_of_total_area"
                    ),
                    # original_hierarchy_df[
                    #     original_hierarchy_df["contour_idx"].isin(underline_contours)
                    # ],
                ]
            )
            .drop_duplicates(subset=["contour_idx"])
            .sort_values("contour_pct_of_image_area", ascending=False)
        )

        # If there *are* underline contours, we'll need to figure out which
        # orientation they are in. We'll look at the midpoint of these contours
        # in relation to the midpoint of the other contours.
        proper_rotation = None
        if underline_contours:
            # Create a DataFrame without any of the underline contours
            non_underline_contours_df = hierarchy_df[
                ~hierarchy_df["contour_idx"].isin(underline_contours)
            ].copy()
            non_underline_mean_x = non_underline_contours_df[
                "contour_midpoint_x"
            ].mean()
            non_underline_mean_y = non_underline_contours_df[
                "contour_midpoint_y"
            ].mean()
            # non_underline_max_x = non_underline_contours_df["contour_midpoint_x"].max()

            # Determine the underline contours
            underline_contours_df = hierarchy_df[
                hierarchy_df["contour_idx"].isin(underline_contours)
            ].copy()
            underline_mean_x = underline_contours_df["contour_midpoint_x"].mean()
            underline_mean_y = underline_contours_df["contour_midpoint_y"].mean()
            # underline_max_x = underline_contours_df["contour_midpoint_x"].max()

            # Determine the range of the x-coordinates and y-coordinates of the underline contours
            underline_contours_xrange = (
                underline_contours_df["contour_midpoint_x"].max()
                - underline_contours_df["contour_midpoint_x"].min()
            )
            underline_contours_yrange = (
                underline_contours_df["contour_midpoint_y"].max()
                - underline_contours_df["contour_midpoint_y"].min()
            )

            # If the yrange is larger than the xrange, then the underlines are either on the left or right side of the tile
            if underline_contours_yrange > underline_contours_xrange:
                # If the mean x-coordinate of the underline contours is less than the mean x-coordinate of the non-underline contours,
                # then the underlines are on the left side of the tile
                if underline_mean_x < non_underline_mean_x:
                    proper_rotation = 270

                # Otherwise, the underlines are on the right side of the tile
                else:
                    proper_rotation = 90

            # Otherwise, the underlines are either on the top or bottom of the tile
            else:
                # If the mean y-coordinate of the underline contours is less than the mean y-coordinate of the non-underline contours,
                # then the underlines are on the top side of the tile
                if underline_mean_y < non_underline_mean_y:
                    proper_rotation = 180

                # Otherwise, the underlines are on the bottom side of the tile
                else:
                    proper_rotation = 0

            special_tile_contours["rotate_fixed"][
                row.tile_sequence_idx
            ] = proper_rotation

        if len(hierarchy_df) == 0:
            raise Exception("No contours were found.")

        # Determine whether the largest level-1 contour is a rectangle
        largest_level_1_contour = contours[hierarchy_df.iloc[0].contour_idx]
        largest_level_1_contour_polygon_approx = approximate_polygon_from_contour(
            largest_level_1_contour, polygon_approximation_epsilon
        )
        largest_level_1_contour_is_rectangle = (
            len(largest_level_1_contour_polygon_approx) == 4
        )

        # If the largest level 1 contour is a rectangle, we're going to determine
        # if it's more of a square or a rectangle
        if largest_level_1_contour_is_rectangle:
            largest_level_1_contour_polygon_approx = (
                largest_level_1_contour_polygon_approx.reshape(-1, 2)
            )

            # Calculate the width and height of the largest level 1 contour
            largest_level_1_contour_width = (
                largest_level_1_contour_polygon_approx[:, 0].max()
                - largest_level_1_contour_polygon_approx[:, 0].min()
            )
            largest_level_1_contour_height = (
                largest_level_1_contour_polygon_approx[:, 1].max()
                - largest_level_1_contour_polygon_approx[:, 1].min()
            )

            # If the width and height are within 10% of each other, then we're going to
            # assume that it's a square
            diff = (
                abs(largest_level_1_contour_width - largest_level_1_contour_height)
                / largest_level_1_contour_width
            )

            if diff <= 0.1:
                special_tile_contours["is_block"].append(row.tile_sequence_idx)
            else:
                special_tile_contours["is_i"].append(row.tile_sequence_idx)

        # We're going to create a new image that only contains the contours
        mask = np.zeros_like(top_down_tile_image)
        mask_with_level_1 = (
            draw_contours(
                img=mask,
                contours=[
                    contours[idx]
                    for idx in list(
                        hierarchy_df.query(
                            "hierarchy_level == 1 and contour_idx not in @underline_contours"
                        ).contour_idx
                    )
                ],
                color=(255, 255, 255),
                thickness=cv2.FILLED,
                return_img=True,
            ),
        )
        mask_with_level_2 = draw_contours(
            img=mask_with_level_1[0],
            contours=[
                contours[idx]
                for idx in list(hierarchy_df.query("hierarchy_level >= 2").contour_idx)
            ],
            color=(0, 0, 0),
            thickness=cv2.FILLED,
            return_img=True,
        )

        # Center the letter image
        centered_letter_img = center_letter_image(
            cv2.cvtColor(mask_with_level_2, cv2.COLOR_BGR2GRAY)
        )

        # Resize the centered letter image
        centered_letter_img = resize_image(centered_letter_img, resize_size)

        # Store the extracted image
        tile_idx_to_extracted_image[row.tile_sequence_idx] = centered_letter_img

    # Return the tile_idx_to_extracted_image dictionary
    return tile_idx_to_extracted_image, special_tile_contours


def display_images_in_grid(images, grid_size=6, convert_to_rgb=True):
    """
    This method will display a list of images in a grid format.
    The method expects the images to be in BGR format (since that's
    what OpenCV uses).
    """

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(5, 5))

    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the index of the cell in the flat list
            index = i * grid_size + j

            # Display the cell in the appropriate subplot
            img_to_show = images[index]
            if convert_to_rgb:
                img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img_to_show)
            axs[i, j].axis("off")  # Hide axis

    plt.show()


# def process_tile_image_tesseract(cur_tile_img, rotation_angles=[0, 90, 180, 270]):
#     """
#     This method will process a single tile image and return some information
#     about the predicted character for that tile. This method will use
#     Tesseract as the character recognition engine.
#     """

#     # Invert the image so that we're working on a white background
#     cur_tile_img = cv2.bitwise_not(cur_tile_img)

#     # Convert the image to a PIL image
#     cur_tile_pil_img = Image.fromarray(cur_tile_img)

#     # Parallelize the processing of the tile image
#     futures = {}
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         for rotation_angle in rotation_angles:
#             # Submit a future to the executor
#             futures[rotation_angle] = executor.submit(
#                 pytesseract.image_to_data,
#                 cur_tile_pil_img.rotate(rotation_angle),
#                 output_type="data.frame",
#                 config="--psm 10 --oem 2 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
#             )

#         # Now, we need to collect the results from the futures
#         results = {}
#         for rotation_angle, future in futures.items():
#             results[rotation_angle] = future.result()

#     # Now that we've collected the results, we'll need to return some information about the
#     # predicted characters for each rotation.
#     dfs_to_concat = []
#     for rotation_angle, df in results.items():
#         df["rotation_angle"] = rotation_angle
#         dfs_to_concat.append(df)
#     result_df = pd.concat(dfs_to_concat)

#     # Drop rows where the text is empty
#     result_df = result_df.dropna(subset=["text"])

#     # Sort by the confidence score
#     result_df = result_df.sort_values(by="conf", ascending=False)

#     # Only include the relevant columns
#     result_df = result_df[["text", "conf", "rotation_angle"]]

#     # Add a column indicating that we'd used Tesseract
#     result_df["method"] = "tesseract"

#     # Divide the confidence score by 100
#     result_df["conf"] = result_df["conf"] / 100

#     # Return the DataFrame
#     return result_df


def thinning(img):
    """
    This method will perform thinning on a binary image. I copied this
    from a ChatGPT conversation: https://chat.openai.com/share/dc6db0ee-6445-4501-a834-7c3c89697bba
    """

    # Convert the image to greyscale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the cross-shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Initialize the thinning image to the max (255)
    thin = np.zeros(img.shape, dtype="uint8")

    # Loop until no more thinning is possible
    iter_ct = 0
    while cv2.countNonZero(img) != 0 and iter_ct < 10:
        # Erode the image
        eroded = cv2.erode(img, element)

        # Open the image
        opened = cv2.dilate(eroded, element)

        # Subtract the opened image from the original image
        temp = cv2.subtract(img, opened)

        # Erode the image, but do not open (dilate) it again
        eroded = cv2.erode(img, element)

        # Store the thinned image
        thin = cv2.bitwise_or(thin, temp)

        # Update the image to the eroded image
        img = eroded.copy()

        # Add to the iteration count
        iter_ct += 1

    return thin


def process_tile_image_easyocr(cur_tile_img, rotation_angles, reader=None):
    """
    This method will process a single tile image and return some information
    about the predicted character for that tile. This method uses EasyOCR as
    the character recognition engine.
    """

    # If the reader is None, we're going to create a new reader
    if reader is None:
        raise Exception("You must specify an EasyOCR reader.")

    # Convert the image to a PIL image
    cur_tile_pil_img = Image.fromarray(cur_tile_img)

    # Parallelize the processing of the tile image
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for rotation_angle in rotation_angles:
            # Submit a future to the executor
            futures[rotation_angle] = executor.submit(
                reader.readtext,
                np.array(cur_tile_pil_img.rotate(rotation_angle)),
                min_size=1,
                text_threshold=0.1,
            )

        # Now, we need to collect the results from the futures
        results = {}
        for rotation_angle, future in futures.items():
            results[rotation_angle] = future.result()

    # Now, we're going to create a DataFrame out of these results
    result_df_records = []
    for rotation_angle, easyocr_results in results.items():
        # Extract the results from the first result
        if len(easyocr_results) == 0:
            boundary_box = None
            text = None
            confidence = None
        else:
            res = easyocr_results[0]
            if len(res) == 0:
                boundary_box = None
                text = None
                confidence = None
            else:
                boundary_box, text, confidence = res

        # Create a record for this result
        result_df_records.append(
            {
                "text": text,
                "conf": confidence,
                "rotation_angle": rotation_angle,
                "method": "easyocr",
            }
        )

    # Finally, make a DataFrame out of the records
    return pd.DataFrame(result_df_records)


def aggregate_prediction_results(result_df, min_prediction_confidence=0.75):
    """
    This method will aggregate the prediction results from the
    multi_engine_tile_processing method. It will return two things:

    1. The predicted tile character
    2. The necessary rotation to make the tile upright

    If the tile is not a character, it will return None for both values
    """

    # If the result_df is just a string, then we're going to return that
    # string (as well as a rotation angle of 0)
    if isinstance(result_df, str):
        return result_df, 0

    # Aggregate the DataFrame
    aggregated_result_df = (
        result_df.query("conf >= @min_prediction_confidence")
        .groupby("text")
        .agg(
            mean_conf=("conf", "mean"),
            pred_ct=("conf", "count"),
            angle_rotations=("rotation_angle", list),
        )
        .reset_index()
    )

    aggregated_result_df["weighted_conf"] = (
        aggregated_result_df["mean_conf"] * aggregated_result_df["pred_ct"]
    )

    aggregated_result_df = aggregated_result_df.sort_values(
        "weighted_conf", ascending=False
    )

    # Only show the letters that're in the allowed Boggle set
    aggregated_result_df = aggregated_result_df[
        aggregated_result_df["text"].isin(settings.allowed_boggle_tiles)
    ]

    # If the length of the DataFrame is 0, then return None
    if len(aggregated_result_df) == 0:
        return None, None

    # Determine the most confident letter prediction
    top_letter_prediction = aggregated_result_df.iloc[0].text

    # Determine the most confident rotation angle for the top letter prediction
    top_rotation_angle = (
        result_df.query("text==@top_letter_prediction")
        .groupby("rotation_angle")
        .agg({"conf": "mean"})
        .reset_index()
        .sort_values("conf", ascending=False)
        .iloc[0]
        .rotation_angle
    )

    return top_letter_prediction, top_rotation_angle


# def ocr_one_tile(
#     tile_img,
#     tile_idx,
#     special_tile_info,
#     engines_to_run=["tesseract", "easyocr"],
#     skeletonize=False,
#     easyocr_reader=None,
# ):
#     """
#     This method will perform optical character recognition on
#     one of the tiles. This method expects three inputs:

#     - `tile_img`: The image of the tile to perform OCR on
#     - `tile_idx`: The index of the tile in the board
#     - `special_tile_info`: A dictionary containing the special
#       tile information for the board. This is used to determine
#     """

#     # First, we're going to check whether or not the tile is a special tile
#     if tile_idx in special_tile_info["is_i"]:
#         return "I"
#     elif tile_idx in special_tile_info["is_block"]:
#         return "BLOCK"

#     # Determine the angles of rotation we'll use when trying to OCR this tile
#     rotation_angles = [0, 90, 180, 270]
#     if tile_idx in special_tile_info["rotate_fixed"]:
#         rotation_angles = [special_tile_info["rotate_fixed"][tile_idx]]

#     # If the user is interested in trying the skeletonized version of the image,
#     # skeletonize the image
#     if skeletonize:
#         skeletonized_img = thinning(tile_img)

#     # We're going to run each of these engines in parallel to speed up the process
#     futures = {}
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         # Now, we're going to iterate through the different OCR engines
#         if "tesseract" in engines_to_run:
#             futures["original_tesseract"] = executor.submit(
#                 process_tile_image_tesseract, tile_img, rotation_angles
#             )

#             # If the user is interested in trying the skeletonized version of the image,
#             # we'll run that as well
#             if skeletonize:
#                 futures["skeletonized_tesseract"] = executor.submit(
#                     process_tile_image_tesseract, skeletonized_img, rotation_angles
#                 )

#         # If the user is interested in trying the easyocr engine, we'll run that as well
#         if "easyocr" in engines_to_run:
#             futures["original_easyocr"] = executor.submit(
#                 process_tile_image_easyocr, tile_img, rotation_angles, easyocr_reader
#             )

#             # Add the skeletonized OCR to the list of futures if the user is interested
#             if skeletonize:
#                 futures["skeletonized_easyocr"] = executor.submit(
#                     process_tile_image_easyocr,
#                     skeletonized_img,
#                     rotation_angles,
#                     easyocr_reader,
#                 )

#         # Now, we need to collect the results from the futures
#         results = {}
#         for rotation_angle, future in futures.items():
#             results[rotation_angle] = future.result()

#     # Now, collect all of the results
#     df_result_list = []
#     for key, future in futures.items():
#         img_type = key.split("_")[0]
#         df_result = future.result()
#         df_result["img_type"] = img_type
#         df_result_list.append(df_result)

#     # Concatenate the results into a single dataframe
#     return pd.concat(df_result_list)


# def ocr_all_tiles(
#     extracted_tile_img_dict,
#     special_tile_info,
#     engines_to_run=["tesseract", "easyocr"],
#     skeletonize=False,
#     easyocr_reader=None,
# ):
#     """
#     This method will OCR all of the tiles. By passing in an `extracted_tile_img_dict` that is
#     keyed with the tile index and has the tile image as the value, we can OCR all of the tiles.

#     This method will return a DataFrame with information about the predicted letter and rotation.
#     """

#     # We're going to process each of the tiles in parallel
#     futures = {}
#     results = {}
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         # Add futures for each tile to the dictionary
#         for tile_idx, tile_img in extracted_tile_img_dict.items():
#             futures[tile_idx] = executor.submit(
#                 ocr_one_tile,
#                 tile_img=tile_img,
#                 tile_idx=tile_idx,
#                 special_tile_info=special_tile_info,
#                 engines_to_run=engines_to_run,
#                 skeletonize=skeletonize,
#                 easyocr_reader=easyocr_reader,
#             )

#         # Now, iterate through all of the futures and wait for them to complete
#         for tile_idx, future in list(futures.items()):
#             results[tile_idx] = aggregate_prediction_results(
#                 future.result(), min_prediction_confidence=0.4
#             )

#     # Finally, we're going to make a DataFrame of all of the tile prediction results
#     result_df = pd.DataFrame(
#         [
#             {
#                 "tile_idx": tile_idx,
#                 "letter": result_list[0],
#                 "rotation_angle": result_list[1],
#             }
#             for tile_idx, result_list in results.items()
#         ]
#     )

#     # Return the result_df
#     return result_df


def parse_boggle_board(
    input_image,
    max_image_height=1500,
    easyocr_reader=None,
    return_parsed_img_sequence=False,
    model=None,
    return_list=None,
):
    """
    This method will run through each of the steps in the
    Boggle board detection process, and then return the
    board as a list of strings.
    """

    # Resize the image to a smaller size
    input_image = resize_image(input_image, max_image_height)

    # STEP 1: LOCATING THE BOARD
    # ====================================================

    # Parameterizing the method
    n_top_contours_to_consider = 200
    min_board_area_threshold = 0.15
    max_board_area_threshold = 0.8
    board_contour_expansion_size = 25
    polygon_approximation_epsilon = 0.05
    binary_threshold_value = 100

    # Detect the board contour
    boggle_board_contour = detect_boggle_board_contour(
        input_image,
        n_top_contours_to_consider,
        min_board_area_threshold,
        max_board_area_threshold,
        board_contour_expansion_size,
        polygon_approximation_epsilon,
        binary_threshold_value,
    )

    # STEP 2: WARPING IMAGE PERPSECTIVE
    # ====================================================

    # Warp the input image to get a top-down view of the board
    top_down_board_image = warp_perspective_to_top_down(
        input_image, boggle_board_contour
    )

    # STEP 3: TILE CONTOUR DETECTION
    # ====================================================

    # Parameterizing the tile detection process
    binary_threshold_value = 100
    min_tile_area_percentage = 0.0003
    max_tile_area_percentage = 0.02
    tile_size_difference_threshold = 0.3
    polygon_approximation_epsilon = 0.02

    # Run the tile detection process
    tile_contours_df = detect_tile_contours(
        top_down_board_image=top_down_board_image,
        binary_threshold_value=binary_threshold_value,
        min_tile_area_percentage=min_tile_area_percentage,
        max_tile_area_percentage=max_tile_area_percentage,
        tile_size_difference_threshold=tile_size_difference_threshold,
    )

    # If "canny_edge_visualization" is in the return_list, we'll return that here
    if "canny_edge_visualization" in return_list:
        canny_edges = apply_canny_edge_detection(
            top_down_board_image,
            resize_height=None,
        )
    # Otherwise, we'll set this to None
    else:
        canny_edges = None

    # STEP 4: TILE IMAGE EXTRACTION
    # ====================================================

    # Parameterize this method
    min_contour_pct_of_total_area = 0.003
    polygon_approximation_epsilon = 0.01
    min_contour_in_underline = 2
    binary_threshold_value = 200
    adaptive_threshold_kernel_size_relative = 0.015
    adaptive_threshold_C = 5
    resize_size = 100

    # Run the tile extraction method
    extracted_tile_img_dict, special_tile_info = extract_tile_images(
        top_down_board_image=top_down_board_image,
        tile_contours_df=tile_contours_df,
        min_contour_pct_of_total_area=min_contour_pct_of_total_area,
        polygon_approximation_epsilon=polygon_approximation_epsilon,
        min_contour_in_underline=min_contour_in_underline,
        adaptive_threshold_kernel_size_relative=adaptive_threshold_kernel_size_relative,
        adaptive_threshold_C=adaptive_threshold_C,
        resize_size=resize_size,
    )

    # If we want to return the parsed image sequence, we'll do that here
    if return_parsed_img_sequence:
        return extracted_tile_img_dict

    # STEP 5: TILE OCR
    # ====================================================

    # Run the OCR on the Boggle tiles
    tile_ocr_results = ocr_all_tiles_cnn(
        extracted_tile_img_dict,
        model,
        return_activation_visualization="activation_visualization" in return_list,
    )

    # Unpack the results of the `ocr_all_tiles_cnn` method
    if "activation_visualization" in return_list:
        tile_ocr_results_df, activation_visualizations = tile_ocr_results
    else:
        tile_ocr_results_df = tile_ocr_results
        activation_visualizations = None

    if return_list is None:
        return tile_ocr_results_df

    # Otherwise, if the user wants to return a list of things, we'll do that here
    return_list_key_to_value = {
        "parsed_board": tile_ocr_results_df,
        "cropped_image": top_down_board_image,
        "tile_contours": tile_contours_df,
        "tile_images": extracted_tile_img_dict,
        "activation_visualization": activation_visualizations,
        "canny_edge_visualization": canny_edges,
    }
    return [return_list_key_to_value[key] for key in return_list]


def ocr_all_tiles_cnn(
    extracted_tile_img_dict,
    model,
    return_activation_visualization=False,
    batch_size=4,
):
    """
    This method will use a specially trainend CNN to run OCR on the tiles.
    """

    # If we want to return the activation visualization, we'll attach the hook
    # that allows us to do that
    if return_activation_visualization:
        # Attach the hook to one of the convolutional layers
        layer_of_interest = model.features[0]
        activated_features = SaveFeatures(layer_of_interest)
    # Create a list of the images and their corresponding indices
    image_list = []
    for image_idx, image in extracted_tile_img_dict.items():
        img_tensor = torch.tensor(image).float()
        image_list.append(img_tensor)

    # Create a DataLoader from the image list
    image_loader = data.DataLoader(
        image_list, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Run the images through the model
    model_predictions = []
    activation_visualizations = []
    for image_batch in image_loader:
        image_batch = image_batch.unsqueeze(1)  # Adds a channel dimension
        score, predicted = torch.max(model(image_batch), 1)
        predicted_letters = [allowed_boggle_tiles[p] for p in predicted.tolist()]
        model_predictions += predicted_letters

        # If we're trying to return the activation visualization, we'll do that here
        if return_activation_visualization:
            # Convert the activations to a numpy array representing a heatmap image
            activation_visualization = activated_features.features
            activation_visualization = np.mean(
                activation_visualization, axis=1
            ).squeeze()
            activation_visualization = np.maximum(activation_visualization, 0)
            activation_visualization /= np.max(activation_visualization)

            # Normalize the numpy array and convert to 8-bit integer
            activation_visualization = (activation_visualization * 255).astype(np.uint8)
            
            # Append each of the activation visualizations to the list
            for idx, viz in enumerate(activation_visualization):
                activation_visualizations.append(viz)

    # Now, we're going to make a DataFrame from the model predictions
    tile_ocr_results_records = []
    for idx, letter in enumerate(model_predictions):
        tile_ocr_results_records.append(
            {
                "tile_idx": idx,
                "letter": letter,
            }
        )
    tile_ocr_results_df = pd.DataFrame(tile_ocr_results_records)

    # If we're not returning the activation visualization, we'll return the tile_ocr_results_df
    if not return_activation_visualization:
        return tile_ocr_results_df

    # Otherwise, we'll return both
    else:
        return tile_ocr_results_df, activation_visualizations
