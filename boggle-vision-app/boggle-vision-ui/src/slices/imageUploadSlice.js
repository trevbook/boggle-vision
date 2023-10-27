import { createSlice } from '@reduxjs/toolkit'

export const imageUploadSlice = createSlice({

    // This slice will be called imageUpload.
    name: "imageUpload",

    // By default, the image will be null. 
    initialState: {
        image: null
    },

    // Define the reducers that'll set the image
    reducers: {

        // This function will set the image
        setImage: (state, action) => {
            state.image = action.payload;
        }

    }

})

// Export the action that'll set the image
export const { setImage } = imageUploadSlice.actions
export default imageUploadSlice.reducer

// // Export a couple of functions that will allow us to open and close the modal.
// export const { openModal, closeModal } = contextModalSlice.actions
// export default contextModalSlice.reducer

