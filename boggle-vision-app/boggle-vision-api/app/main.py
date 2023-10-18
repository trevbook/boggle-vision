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


# ============================================
#                 ENDPOINTS
# ============================================
# Below, we'll define all of the endpoints for the API.


@app.get("/")
def read_root():
    print("Hey.")
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


from fastapi import FastAPI
from typing import List

@app.post("/solve_board")
def solve_board(board_data: List[str]):
    """
    The following method will solve a Boggle board 
    when given a list of letters that it contains.
    """
    
    # Parse a 2D board matrix from the board data
    board_matrix = board_solve.parse_board_from_letter_sequence([x.lower() for x in board_data])
    
    # Solve the board
    solved_boggle_board_df = board_solve.solve_boggle(board_matrix, board_solve.allowed_words_trie)
    
    # Return the solved board as a records-style JSON
    return solved_boggle_board_df.to_dict(orient="records")

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
    parsed_board_df = board_detect.parse_boggle_board(
        input_image=image,
        max_image_height=1200,
        model=net
    )
    
    # Get the letter sequence
    letter_sequence = list(parsed_board_df["letter"])

    # Now, we're going to run the image through the CNN
    return {
        "letter_sequence": letter_sequence
    }

