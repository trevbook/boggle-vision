import { createSlice } from "@reduxjs/toolkit";

export const boardDataSlice = createSlice({
  // This slice will be called boardData.
  name: "boardData",

  // By default, the board data
  initialState: {
    boardData: null,
    wordsTableData: null,
    boardStats: null,
    letterSequence: null,
    n_words: null
  },

  // Define the reducers that'll deal with the board data
  reducers: {
    // This function will set the board data
    setBoardData: (state, action) => {
      state.boardData = action.payload;
    },

    // This function will set the words table data
    setWordsTableData: (state, action) => {
      state.wordsTableData = action.payload;
      state.n_words = action.payload.length
    },

    // This function will set the board stats
    setBoardStats: (state, action) => {
      state.boardStats = action.payload;
    },

    // This function will set the letter sequence
    setLetterSequence: (state, action) => {
      state.letterSequence = action.payload;
    },
  },
});

// Export the action that'll set the image
export const {
  setBoardData,
  setWordsTableData,
  setBoardStats,
  setLetterSequence,
} = boardDataSlice.actions;
export default boardDataSlice.reducer;
