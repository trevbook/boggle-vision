import { createSlice } from "@reduxjs/toolkit";

export const imageUploadSlice = createSlice({
  // This slice will be called imageUpload.
  name: "imageUpload",

  // By default, the image will be null.
  initialState: {
    image: null,
    loading: false,
  },

  // Define the reducers that'll set the image
  reducers: {
    // This function will set the image
    setImage: (state, action) => {
      state.image = action.payload;
    },

    // This reducer will set the loading state
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
  },
});

// Export the action that'll set the image
export const { setImage, setLoading } = imageUploadSlice.actions;
export default imageUploadSlice.reducer;
