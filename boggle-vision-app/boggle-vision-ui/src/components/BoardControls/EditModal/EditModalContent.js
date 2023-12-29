// This component is EditModalContent, which will contain the content for a modal that allows the user to edit the board.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  setBoardData,
  setLetterSequence,
} from "../../../slices/boardDataSlice";
import { Button, Grid, Select } from "@mantine/core";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the EditModalContent.
 */
const EditModalContent = () => {
  // ==================
  // STATE MANAGEMENT
  // ==================

  // Set up a dispatch.
  const dispatch = useDispatch();

  // Set up a selector for the letterSequence from the boardData.
  const letterSequence = useSelector((state) => state.boardData.letterSequence);

  // This selector will contain the boardData.
  const boardData = useSelector((state) => state.boardData.boardData);

  // Set up a state for the curBoardLetters.
  const [curBoardLetters, setCurBoardLetters] = useState(null);

  // This state will keep track of the size of the grid
  const [gridSize, setGridSize] = useState(16); // Default to 4x4

  // This variable contains all of the allowed letters
  const allowedLetters = [
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
    "BLOCK",
  ];

  // ==================
  // HELPER FUNCTIONS
  // ==================

  // This function will help to determine the board size
  const getBoardSize = (lettersCount) => {
    const sizes = {
      16: "Boggle (4x4)",
      25: "Big Boggle (5x5)",
      36: "Super Big Boggle (6x6)",
    };
    return sizes[lettersCount] || sizes[16]; // Default to 4x4 if size is unknown
  };

  // This function will handle a change in the size of the board
  const handleBoardSizeChange = (value) => {
    // TODO: Implement this function
  };

  // Function to handle letter change in the dropdown
  const handleLetterChange = (index, value) => {
    const updatedLetters = [...curBoardLetters];
    updatedLetters[index] = value;
    setCurBoardLetters(updatedLetters);
  };

  // Validate the current board letters
  const validateInput = () => {
    // Check if all current board letters are within the allowed letters
    return curBoardLetters.every((letter) => allowedLetters.includes(letter));
  };

  // Function to handle submit
  const handleSubmit = () => {
    //
    console.log(`The submit button has been clicked.`);
    console.log(`We're going to update the board data with the following:`);
    console.log(curBoardLetters);

    // Update the letterSequence in the boardDataSlice
    dispatch(setLetterSequence(curBoardLetters));

    // Update the boardData in the boardDataSlice, changing the letter_sequence attribute
    const updatedBoardData = { ...boardData };
    updatedBoardData.letter_sequence = curBoardLetters;
    dispatch(setBoardData(updatedBoardData));
  };

  // ==================
  // COMPONENT EFFECTS
  // ==================

  // This effect will be called when the board data changes, and will extract the `letter_sequence`
  // attribute from the board data.
  useEffect(() => {
    // If board_data is null or doesnt have the letter_sequence attribute, then return.
    if (letterSequence === null) {
      return;
    }

    // If the size of the letter_sequence is not 16, 25, or 36, then return.
    if (![16, 25, 36].includes(letterSequence.length)) {
      return;
    }

    // Set the current board letters to the letter sequence.
    setCurBoardLetters(letterSequence);

    // Determine the grid size (which is the square root of the length of the letter sequence)
    const gridSize = Math.sqrt(letterSequence.length);
    setGridSize(gridSize);
  }, [JSON.stringify(letterSequence)]);

  // ==================
  // RENDERING
  // ==================

  // Render the component
  return (
    <div>
      {curBoardLetters === null ? (
        <></>
      ) : (
        <div>
          {/* Board Size Dropdown */}
          <div style={{ marginBottom: "15px" }}>
            <Select
              label="Board Size"
              value={getBoardSize(curBoardLetters.length)}
              onChange={handleBoardSizeChange}
              data={[
                "Boggle (4x4)",
                "Big Boggle (5x5)",
                "Super Big Boggle (6x6)",
              ]}
            />
          </div>

          <div style={{ marginBottom: "15px" }}>
            {/* Letter Grid */}
            <Grid>
              {curBoardLetters.map((letter, index) => (
                <Grid.Col
                  span={12 / gridSize} // for desktop
                  xs={2} // for very small screens like mobile
                  sm={4} // for small screens like tablets
                  md={6} // for medium screens
                  key={index}
                >
                  <input
                    type="text"
                    value={letter}
                    onChange={(e) => handleLetterChange(index, e.target.value)}
                    style={{
                      textAlign: "center",
                      border: allowedLetters.includes(letter)
                        ? "1px solid black"
                        : "2px solid red",
                      color: "black",
                      backgroundColor: "white",
                      width: "100%", // make input responsive
                      boxSizing: "border-box", // include padding and borders in the element's total width and height
                    }}
                    maxLength={2} // Adjust for 'Qu' tile if needed
                  />
                </Grid.Col>
              ))}
            </Grid>
          </div>
          <div style={{ marginBottom: "15px" }}>
            {/* Submit Button */}
            <Button fullWidth onClick={handleSubmit}>
              Submit
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

// Export the component.
export default EditModalContent;
