import { configureStore } from "@reduxjs/toolkit";
import imageUploadSlice from "./slices/imageUploadSlice";
import boardDataSlice from "./slices/boardDataSlice";

export default configureStore({
    reducer: {
        imageUpload: imageUploadSlice,
        boardData: boardDataSlice
    },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware({
        serializableCheck: false
    })
})