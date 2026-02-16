import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import math
from matplotlib import pyplot as plt
import numpy as np
from statistics import mode
import cv2
# import pytesseract
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import easyocr
import traceback
import settings

# Indicate which tiles are allowed in Boggle
allowed_boggle_tiles = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "Qu",
    "Er",
    "Th",
    "In",
    "An",
    "He",
]


def build_hierarchy_tree(hierarchy):
    tree = defaultdict(list)
    root_nodes = []

    # Build the hierarchy tree
    for i, (_, _, _, parent) in enumerate(hierarchy[0]):
        if parent == -1:
            root_nodes.append(i)
        else:
            tree[parent].append(i)

    return tree, root_nodes


def assign_levels(tree, root_nodes):
    levels = {}
    stack = [(node, 0) for node in root_nodes]  # (node, level)

    # Depth-first search
    while stack:
        node, level = stack.pop()
        levels[node] = level
        for child in tree[node]:
            stack.append((child, level + 1))

    return levels


def display_rainbow_hierarchy(input_image):
    # Convert to greyscale and apply thresholding
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    ret, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Detect the contours and hierarchy
    contours, hierarchy = cv2.findContours(
        thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Define rainbow colors using the BGR format
    rainbow_colors = [
        (0, 0, 255),
        (0, 127, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
        (130, 0, 75),
    ]

    # Build the hierarchy tree and assign levels to each contour
    tree, root_nodes = build_hierarchy_tree(hierarchy)
    levels = assign_levels(tree, root_nodes)

    # Create an image copy to draw on
    image_copy = input_image.copy()

    # Draw each contour with a color based on its level
    for contour, level in levels.items():
        if level < 6:  # Only consider up to 6 levels
            color = rainbow_colors[level]
            cv2.drawContours(image_copy, [contours[contour]], -1, color, 2)

    # Display the image using matplotlib
    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)


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


def hierarchy_to_dataframe(hierarchy):
    """
    This function will convert the hierarchy produced by cv2.findContours into a pandas dataframe.
    The DataFrame will have information about the contour index, hierarchy level, parent contour index,
    and children contour indices.
    """

    def dfs(contour_idx, level):
        """
        This is a helper method that performs a depth-first search on the contour hierarchy.
        """
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
            dfs(child, level + 1)

    data = []
    root_contours = np.where(hierarchy[0, :, 3] == -1)[0]

    # Add siblings of root contours
    for contour_idx in root_contours:
        next_sibling, prev_sibling, _, _ = hierarchy[0, contour_idx]
        while next_sibling != -1:
            root_contours = np.append(root_contours, next_sibling)
            next_sibling, _, _, _ = hierarchy[0, next_sibling]
        while prev_sibling != -1:
            root_contours = np.append(root_contours, prev_sibling)
            _, prev_sibling, _, _ = hierarchy[0, prev_sibling]

    # Run DFS from each root contour
    for contour_idx in root_contours:
        dfs(contour_idx, 0)

    df = pd.DataFrame(
        data, columns=["contour_idx", "hierarchy_level", "parent", "children"]
    )

    return df.drop_duplicates(subset=["contour_idx"])


# def process_tile_image_tesseract(cur_tile_img):
#     """
#     This method will process a single tile image and return some information
#     about the predicted character for that tile. This method will use
#     Tesseract as the character recognition engine.
#     """

#     # Make the image greyscale
#     # cur_tile_img = cv2.cvtColor(cur_tile_img, cv2.COLOR_BGR2GRAY)

#     # Invert the image
#     cur_tile_img = cv2.bitwise_not(cur_tile_img)

#     # Convert the image to a PIL image
#     cur_tile_pil_img = Image.fromarray(cur_tile_img)

#     # Parallelize the processing of the tile image
#     futures = {}
#     with ThreadPoolExecutor() as executor:
#         for rotation_angle in range(0, 360, 90):
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


def process_tile_image_easyocr(cur_tile_img, reader):
    """
    This method will process a single tile image and return some information
    about the predicted character for that tile. This method uses EasyOCR as
    the character recognition engine.
    """

    # Convert the image to a PIL image
    cur_tile_pil_img = Image.fromarray(cur_tile_img)

    # Parallelize the processing of the tile image
    futures = {}
    with ThreadPoolExecutor() as executor:
        for rotation_angle in range(0, 360, 90):
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


def thinning(img):
    """
    This method will perform thinning on a binary image. I copied this
    from a ChatGPT conversation: https://chat.openai.com/share/dc6db0ee-6445-4501-a834-7c3c89697bba
    """

    # Convert the image to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


# def multi_engine_tile_processing(cur_tile_img, reader):
#     """
#     This method will run through OCR on the tile using a
#     variety of different methods (Tesseract, EasyOCR) and
#     a couple of different image processing techniques (base-image, thinning).

#     It'll return a DataFrame with all of the predictions and their confidence levels.
#     """

#     # Use the thinning function to get the skeleton of the image
#     cur_tile_img_skeleton = thinning(cur_tile_img)

#     # Execute the character recognition in parallel
#     futures = {}
#     with ThreadPoolExecutor() as executor:
#         futures["original_img_tesseract"] = executor.submit(
#             process_tile_image_tesseract, cur_tile_img
#         )
#         futures["skeleton_img_tesseract"] = executor.submit(
#             process_tile_image_tesseract, cur_tile_img_skeleton
#         )
#         futures["original_img_easyocr"] = executor.submit(
#             process_tile_image_easyocr, cur_tile_img, reader
#         )
#         futures["skeleton_img_easyocr"] = executor.submit(
#             process_tile_image_easyocr, cur_tile_img_skeleton, reader
#         )

#     # Now, collect all of the results
#     df_result_list = []
#     for key, future in futures.items():
#         img_type = key.split("_")[0]
#         df_result = future.result()
#         df_result["img_type"] = img_type
#         df_result_list.append(df_result)

#     # Concatenate the results into a single dataframe
#     return pd.concat(df_result_list)


# def aggregate_prediction_results(result_df, min_prediction_confidence=0.75):
#     """
#     This method will aggregate the prediction results from the
#     multi_engine_tile_processing method. It will return two things:

#     1. The predicted tile character
#     2. The necessary rotation to make the tile upright

#     If the tile is not a character, it will return None for both values
#     """

#     # Aggregate the DataFrame
#     aggregated_result_df = (
#         result_df.query("conf >= @min_prediction_confidence")
#         .groupby("text")
#         .agg(
#             mean_conf=("conf", "mean"),
#             pred_ct=("conf", "count"),
#             angle_rotations=("rotation_angle", list),
#         )
#         .reset_index()
#     )

#     aggregated_result_df["weighted_conf"] = (
#         aggregated_result_df["mean_conf"] * aggregated_result_df["pred_ct"]
#     )

#     aggregated_result_df = aggregated_result_df.sort_values(
#         "weighted_conf", ascending=False
#     )

#     # Only show the letters that're in the allowed Boggle set
#     aggregated_result_df = aggregated_result_df[
#         aggregated_result_df["text"].isin(allowed_boggle_tiles)
#     ]

#     # If the length of the DataFrame is 0, then return None
#     if len(aggregated_result_df) == 0:
#         return None, None

#     # Determine the most confident letter prediction
#     top_letter_prediction = aggregated_result_df.iloc[0].text

#     # Determine the most confident rotation angle for the top letter prediction
#     top_rotation_angle = (
#         result_df.query("text==@top_letter_prediction")
#         .groupby("rotation_angle")
#         .agg({"conf": "mean"})
#         .reset_index()
#         .sort_values("conf", ascending=False)
#         .iloc[0]
#         .rotation_angle
#     )

#     return top_letter_prediction, top_rotation_angle


def identify_underline_contours(input_img, input_hierarchy_df):
    """
    This function will take in a modified contour hierarchy DataFrame, and return a list
    of the contours that're suspected to be underlines. This is helpful in detecting
    W's, M's, and Z's, which are underlined in Boggle.
    """
    try:
        # Filter out any contours that're larger than ~3% of the image
        input_hierarchy_df = input_hierarchy_df.query(
            "pct_of_total_area <= 0.03 & pct_of_total_area >= 0.001"
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
        if len(underline_contour_df) < 3:
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


def draw_contours(img, contours, color=(0, 255, 0), thickness=3):
    """
    This method will show the contours on an image.
    """

    img_copy = img.copy()
    img_copy = cv2.drawContours(
        img_copy,
        # contours=[polygon],
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
    )

    plt.imshow(img_copy)
    plt.show()
