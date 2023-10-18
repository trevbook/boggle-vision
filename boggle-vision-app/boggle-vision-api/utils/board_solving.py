# This file contains utils that help with solving a Boggle board.

# ============================================
#                    SETUP
# ============================================
# The code below will set up the rest of this file.

# Import statements
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Import custom-built modules
from utils.settings import allowed_boggle_tiles

# We're going to load in the trie that we created in the
# "Solving a Boggle Board" notebook
with open("utils/allowed-words-trie.json", "r") as json_file:
    allowed_words_trie = json.load(json_file)

# ============================================
#                    METHODS
# ============================================
# The methods below will be used to solve a Boggle board.


def parse_board_from_letter_sequence(letter_sequence):
    """
    This method will parse a Boggle board (in the form of a 2D list)
    from a sequence of semi-colon delimited letters.
    """
    board = []
    n_rows = int(len(letter_sequence) ** 0.5)
    for i in range(n_rows):
        board.append(letter_sequence[i * n_rows : (i + 1) * n_rows])
    return board


def score_boggle_word(word):
    """
    This method will score a Boggle word.
    """

    if len(word) < 4:
        return 0
    elif len(word) == 4:
        return 1
    elif len(word) == 5:
        return 2
    elif len(word) == 6:
        return 3
    elif len(word) == 7:
        return 5
    else:
        return 11


# Check if a position is within the Boggle board
def is_valid(x, y, visited, board):
    return 0 <= x < len(board) and 0 <= y < len(board[0]) and not visited[x][y]


# Run DFS on the board to find words
def dfs(x, y, board, visited, trie, word, found_words):
    if trie.get("end"):
        found_words.add(word)

    visited[x][y] = True

    # Adjacent positions (up, down, left, right, and diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if is_valid(new_x, new_y, visited, board) and board[new_x][new_y] != "block":
            next_char = board[new_x][new_y]
            if next_char in trie:
                dfs(
                    new_x,
                    new_y,
                    board,
                    visited,
                    trie[next_char],
                    word + next_char,
                    found_words,
                )

    visited[x][y] = False


# Solve the Boggle board
def solve_boggle(board, trie):
    found_words = set()

    rows, cols = len(board), len(board[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            start_char = board[i][j]
            if start_char in trie:
                dfs(
                    i,
                    j,
                    board,
                    visited,
                    trie[start_char],
                    start_char,
                    found_words,
                )

    # Now, we're going to create a DataFrame that contains the words found on each board
    available_words_df = pd.DataFrame.from_records(
        [
            {"word": word, "length": len(word), "points": score_boggle_word(word)}
            for word in found_words
        ]
    )
    
    if len(available_words_df) > 0:
        available_words_df = available_words_df.sort_values(
            "length", ascending=False
        ).query("length >= 4")

    return available_words_df
