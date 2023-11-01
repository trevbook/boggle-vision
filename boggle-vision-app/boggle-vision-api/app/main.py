# This is the main entrypoint to the Boggle Vision API.

# ============================================
#                    SETUP
# ============================================
# The code below will set up the rest of this file.

# Import statements
import torch
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image
import io
import json
from typing import List

# from matplotlib.colors import rgb2hex

# Importing some custom modules
import utils.board_detection as board_detect
import utils.board_solving as board_solve
from utils.cnn import BoggleCNN

# Setting up the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "*", "http://192.168.1.159:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load in the CNN model
net = BoggleCNN()
net.load_state_dict(torch.load("models/boggle_cnn.pth"))

# Load in the Boggle board point distribution
with open("data/boggle-board-point-stats.json", "r") as json_file:
    boggle_board_point_stats = json.load(json_file)


def hex_to_rgb(hex_color):
    """Converts a hex color code to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# Assuming rgb2hex is similar to your original implementation but adapted for [0, 255] range
def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def get_custom_gradient_color(
    board_score, board_stats, red="#750701", yellow="#8c8c00", green="#035903"
):
    """
    Determines a custom continuous color for the total number of points on a Boggle board.

    Parameters:
    - board_score (float): The total number of points on the board
    - board_stats (dict): Statistics about the distribution of total points across various Boggle boards
    - red (str): Hex code for the red color
    - yellow (str): Hex code for the yellow color
    - green (str): Hex code for the green color

    Returns:
    - str: The hex color code to represent the board_score
    """
    # Extract relevant percentiles and min/max values from the board_stats
    min_score = board_stats["min"]
    first_quartile = board_stats["25%"]
    third_quartile = board_stats["75%"]
    max_score = board_stats["max"]

    # Convert hex codes to RGB
    red = hex_to_rgb(red)
    yellow = hex_to_rgb(yellow)
    green = hex_to_rgb(green)

    # Calculate the interpolated color based on board_score
    if board_score <= first_quartile:
        t = (board_score - min_score) / (first_quartile - min_score)
        color = [(1 - t) * red[i] + t * yellow[i] for i in range(3)]
    elif board_score <= third_quartile:
        t = (board_score - first_quartile) / (third_quartile - first_quartile)
        color = [(1 - t) * yellow[i] + t * green[i] for i in range(3)]
    else:
        t = (board_score - third_quartile) / (max_score - third_quartile)
        color = [
            (1 - t) * green[i] + t * green[i] for i in range(3)
        ]  # Remain green as score goes higher

    # Convert RGB to hex color code
    hex_color = rgb2hex(color)
    return hex_color


# ============================================
#                 ENDPOINTS
# ============================================
# Below, we'll define all of the endpoints for the API.


@app.post("/solve_board")
def solve_board(board_data: List[str]):
    """
    The following method will solve a Boggle board
    when given a list of letters that it contains.
    """

    # Parse a 2D board matrix from the board data
    board_matrix = board_solve.parse_board_from_letter_sequence(
        [x.lower() for x in board_data]
    )

    # Solve the board
    solved_boggle_board_df = board_solve.solve_boggle(
        board_matrix, board_solve.allowed_words_trie
    )

    # Create a dictionary mapping the word_id to the path
    word_id_to_path_dict = {
        row.word_id: row.path for row in solved_boggle_board_df.itertuples()
    }

    # Determine the total number of points in the solved_boggle_board_df
    total_points = int(solved_boggle_board_df["points"].sum())

    # Determine the number of 11-point words
    num_eleven_point_words = int((solved_boggle_board_df["points"] == 11).sum())

    # Determine the number of words
    num_words = len(solved_boggle_board_df)

    # Determine the length of the longest word
    longest_word_length = int(solved_boggle_board_df["length"].max())

    # Determine the word with the longest length
    longest_word = solved_boggle_board_df.sort_values("length", ascending=False).iloc[
        0
    ]["word"]

    # Unpack the boggle_board_point_stats
    boggle_board_total_points_stats = boggle_board_point_stats.get("total_points", {})
    boggle_board_eleven_pointers_stats = boggle_board_point_stats.get(
        "eleven_pointers", {}
    )
    boggle_board_word_count_stats = boggle_board_point_stats.get("word_count", {})

    # Get the color for the total number of points
    total_points_color = get_custom_gradient_color(
        total_points, boggle_board_total_points_stats
    )
    eleven_pointers_color = get_custom_gradient_color(
        num_eleven_point_words, boggle_board_eleven_pointers_stats
    )
    word_count_color = get_custom_gradient_color(
        num_words, boggle_board_word_count_stats
    )
    avg_points_per_word_color = get_custom_gradient_color(
        total_points / num_words, boggle_board_point_stats["avg_points_per_word"]
    )

    # Determine how many standard deviations the total_points is from the mean
    total_points_mean = boggle_board_total_points_stats["mean"]
    total_points_std = boggle_board_total_points_stats["std"]
    total_points_z_score = (total_points - total_points_mean) / total_points_std

    # Determine how many standard deviations the num_eleven_point_words is from the mean
    eleven_pointers_mean = boggle_board_eleven_pointers_stats["mean"]
    eleven_pointers_std = boggle_board_eleven_pointers_stats["std"]
    eleven_pointers_z_score = (
        num_eleven_point_words - eleven_pointers_mean
    ) / eleven_pointers_std

    # Determine how many standard deviations the num_words is from the mean
    word_count_mean = boggle_board_word_count_stats["mean"]
    word_count_std = boggle_board_word_count_stats["std"]
    word_count_z_score = (num_words - word_count_mean) / word_count_std

    # Determine the average points per word
    avg_points_per_word = total_points / num_words

    # Determine how many standard deviations the avg_points_per_word is from the mean
    avg_points_per_word_mean = boggle_board_point_stats["avg_points_per_word"]["mean"]
    avg_points_per_word_std = boggle_board_point_stats["avg_points_per_word"]["std"]
    avg_points_per_word_z_score = (
        avg_points_per_word - avg_points_per_word_mean
    ) / avg_points_per_word_std

    # Return the solved board as a records-style JSON
    return {
        "board_stats": {
            "total_points": total_points,
            "eleven_point_words": num_eleven_point_words,
            "word_count": num_words,
            "longest_word_length": longest_word_length,
            "longest_word": longest_word,
            "total_points_color": total_points_color,
            "eleven_pointers_color": eleven_pointers_color,
            "word_count_color": word_count_color,
            "total_points_z_score": total_points_z_score,
            "eleven_pointers_z_score": eleven_pointers_z_score,
            "word_count_z_score": word_count_z_score,
            "avg_points_per_word": float(f"{avg_points_per_word:.2f}"),
            "avg_points_per_word_color": avg_points_per_word_color,
            "avg_points_per_word_z_score": avg_points_per_word_z_score,
        },
        "solved_board": solved_boggle_board_df.to_dict(orient="records"),
        "word_id_to_path": word_id_to_path_dict
    }


# The following endpoint is a POST endpoint that will
# take in data and then return the JSON representation
# of that data.
@app.post("/analyze_image")
def analyze_image(data: dict):
    # Extract the image string from the dict
    image_str = data.get("image")
    pure_base64_str = image_str.split(",")[-1]

    # Pad the base64 string if it's not a multiple of 4
    padded_image_str = pure_base64_str + "=" * ((4 - len(pure_base64_str) % 4) % 4)

    # Decode the base64 string
    image_bytes = base64.b64decode(padded_image_str)

    # Load bytes into a color image using cv2
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), -1)

    # Parse the board
    (
        parsed_board_df,
        cropped_board_img,
        tile_contours_df,
    ) = board_detect.parse_boggle_board(
        input_image=image,
        max_image_height=1200,
        model=net,
        return_list=["parsed_board", "cropped_image", "tile_contours"],
    )

    # Determine the width and height of the cropped_board_img
    cropped_board_height, cropped_board_width = cropped_board_img.shape[:2]

    # Encode the image as a base64 string
    _, buffer = cv2.imencode(".png", cropped_board_img)
    cropped_board_img_str = base64.b64encode(buffer.tobytes()).decode("utf-8")

    # Get the letter sequence
    letter_sequence = list(parsed_board_df["letter"])

    # Make a dictionary mapping tile index to contour
    tile_idx_to_contour_dict = {
        row.tile_sequence_idx: row.contour.squeeze().tolist()
        for row in tile_contours_df.itertuples()
    }

    # Now, we're going to run the image through the CNN
    return {
        "letter_sequence": letter_sequence,
        "cropped_board": cropped_board_img_str,
        "tile_contours": tile_idx_to_contour_dict,
        "cropped_board_width": cropped_board_width,
        "cropped_board_height": cropped_board_height,
    }
