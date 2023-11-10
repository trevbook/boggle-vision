// Import statements
import { createSlice } from "@reduxjs/toolkit";

export const boardImagesSlice = createSlice({
  // This slice will be called boardImages.
  name: "boardImages",

  // By default, the board data
  initialState: {
    boardImages: null,
    letterImageContours: null,
    boardImageOriginalHeight: null,
    boardImageOriginalWidth: null,
    wordToLetterContourPath: null,
    letterImageActivations: null,
  },

  // Define the reducers that'll deal with the board data
  reducers: {
    // This function will set the letter image contours
    setLetterImageContours: (state, action) => {
      state.letterImageContours = action.payload;
    },

    // This function will set the board data
    setBoardImages: (state, action) => {
      state.boardImages = action.payload;
    },

    setBoardImageOriginalHeight: (state, action) => {
      state.boardImageOriginalHeight = action.payload;
    },

    setBoardImageOriginalWidth: (state, action) => {
      state.boardImageOriginalWidth = action.payload;
    },

    setWordToLetterContourPath: (state, action) => {
      state.wordToLetterContourPath = action.payload;
    },

    setLetterImageActivations: (state, action) => {
      state.letterImageActivations = action.payload;
    }
  },
});

// Export the action that'll set the image
export const {
  setBoardImages,
  setLetterImageContours,
  setBoardImageOriginalHeight,
  setBoardImageOriginalWidth,
  setWordToLetterContourPath,
  setLetterImageActivations
} = boardImagesSlice.actions;
export default boardImagesSlice.reducer;
