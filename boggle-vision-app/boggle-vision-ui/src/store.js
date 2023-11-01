import { configureStore } from "@reduxjs/toolkit";
import imageUploadSlice from "./slices/imageUploadSlice";
import boardDataSlice from "./slices/boardDataSlice";
import boardImagesSlice from "./slices/boardImagesSlice";
import userControlSlice from "./slices/userControlSlice";

export default configureStore({
    reducer: {
        imageUpload: imageUploadSlice,
        boardData: boardDataSlice,
        boardImages: boardImagesSlice,
        userControl: userControlSlice
    },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware({
        serializableCheck: false
    })
})