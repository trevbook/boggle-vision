import { configureStore } from "@reduxjs/toolkit";
import imageUploadSlice from "./slices/imageUploadSlice";
import boardDataSlice from "./slices/boardDataSlice";
import boardImagesSlice from "./slices/boardImagesSlice";

export default configureStore({
    reducer: {
        imageUpload: imageUploadSlice,
        boardData: boardDataSlice,
        boardImages: boardImagesSlice
    },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware({
        serializableCheck: false
    })
})