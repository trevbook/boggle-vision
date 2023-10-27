// Import statements
import { createSlice } from '@reduxjs/toolkit'

export const boardImagesSlice = createSlice({

    // This slice will be called boardImages.
    name: "boardImages",

    // By default, the board data
    initialState: {
        boardImages: null
    },

    // Define the reducers that'll deal with the board data
    reducers: {

        // This function will set the board data
        setBoardImages: (state, action) => {
            state.boardImages = action.payload;
        }

    }

})

// Export the action that'll set the image
export const { setBoardImages } = boardImagesSlice.actions
export default boardImagesSlice.reducer
