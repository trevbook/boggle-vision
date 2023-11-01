import { createSlice } from "@reduxjs/toolkit";

export const userControlSlice = createSlice({
  // This slice will be called userControl.
  name: "userControl",

  // Define the initial state of this slice
  initialState: {
    show_letter_overlay: false,
    selected_word_index: null
  },

  // Define the reducers for this slice
  reducers: {
    
    // This reducer will toggle the show_letter_overlay state
    toggleLetterOverlay: (state, action) => {
      state.show_letter_overlay = !state.show_letter_overlay;
    },

    // This reducer will set the selected_word_index state
    setSelectedWordIndex: (state, action) => {
      console.log(`Setting selected word index to ${action.payload}`);
      state.selected_word_index = action.payload;
    }

  },
});

// Export this slice's different actions
export const { toggleLetterOverlay, setSelectedWordIndex } = userControlSlice.actions;
export default userControlSlice.reducer;
