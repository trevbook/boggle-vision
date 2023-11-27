import { createSlice } from "@reduxjs/toolkit";

export const userControlSlice = createSlice({
  // This slice will be called userControl.
  name: "userControl",

  // Define the initial state of this slice
  initialState: {
    show_letter_overlay: false,
    selected_word_index: null,
    current_visual_filter: null,
    feature_activations_filter_primary_color: "#ffffff",
    feature_activations_filter_secondary_color: "#000000",
    canny_edge_filter_primary_color: "#ffffff",
    canny_edge_filter_secondary_color: "#000000"
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

    // This reducer will set the feature_activations_filter_primary_color state
    setFeatureActivationsFilterPrimaryColor: (state, action) => {
      state.feature_activations_filter_primary_color = action.payload;
    },

    // This reducer will set the feature_activations_filter_secondary_color state
    setFeatureActivationsFilterSecondaryColor: (state, action) => {
      state.feature_activations_filter_secondary_color = action.payload;
    },

    // This reducer will set the canny_edge_filter_primary_color state
    setCannyEdgeFilterPrimaryColor: (state, action) => {
      state.canny_edge_filter_primary_color = action.payload;
    },

    // This reducer will set the canny_edge_filter_secondary_color state
    setCannyEdgeFilterSecondaryColor: (state, action) => {
      state.canny_edge_filter_secondary_color = action.payload;
    },

    // This reducer will reset all of the user controls
    resetAllControls: (state, action) => {
      state.show_letter_overlay = false;
      state.selected_word_index = null;
      state.current_visual_filter = null;
      state.feature_activations_filter_primary_color = "#ffffff";
      state.feature_activations_filter_secondary_color = "#000000";
      state.canny_edge_filter_primary_color = "#ffffff";
      state.canny_edge_filter_secondary_color = "#000000";
    },
  },
});

// Export this slice's different actions
export const {
  toggleLetterOverlay,
  setSelectedWordIndex,
  setCurrentVisualFilter,
  setFeatureActivationsFilterPrimaryColor,
  setFeatureActivationsFilterSecondaryColor,
  setCannyEdgeFilterPrimaryColor,
  setCannyEdgeFilterSecondaryColor,
  resetAllControls,
} = userControlSlice.actions;
export default userControlSlice.reducer;
