import { createSlice } from '@reduxjs/toolkit'

export const boardDataSlice = createSlice({

    // This slice will be called boardData.
    name: "boardData",

    // By default, the board data
    initialState: {
        boardData: null,
        wordsTableData: null,
        boardStats: null
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
        },

        // This function will set the board stats
        setBoardStats: (state, action) => {
            state.boardStats = action.payload;
        }

    }

})

// Export the action that'll set the image
export const { setBoardData, setWordsTableData, setBoardStats } = boardDataSlice.actions
export default boardDataSlice.reducer
