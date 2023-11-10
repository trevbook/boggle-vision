import { createSlice } from "@reduxjs/toolkit";

export const userControlSlice = createSlice({
  // This slice will be called userControl.
  name: "userControl",

  // Define the initial state of this slice
  initialState: {
    show_letter_overlay: false,
    selected_word_index: null,
    current_visual_filter: null,
  },

  // Define the reducers for this slice
  reducers: {
    // This reducer will toggle the show_letter_overlay state
    toggleLetterOverlay: (state, action) => {
      state.show_letter_overlay = !state.show_letter_overlay;
    },

    // This reducer will set the selected_word_index state
    setSelectedWordIndex: (state, action) => {
      state.selected_word_index = action.payload;
    },

    // This reducer will set the current_visual_filter state
    setCurrentVisualFilter: (state, action) => {
      state.current_visual_filter = action.payload;
    },

    resetAllControls: (state, action) => {
      state.show_letter_overlay = false;
      state.selected_word_index = null;
      state.current_visual_filter = null;
    }
  },
});

// Export this slice's different actions
export const {
  toggleLetterOverlay,
  setSelectedWordIndex,
  setCurrentVisualFilter,
  resetAllControls
} = userControlSlice.actions;
export default userControlSlice.reducer;
