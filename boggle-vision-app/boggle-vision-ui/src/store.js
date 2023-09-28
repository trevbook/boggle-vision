import { configureStore } from "@reduxjs/toolkit";
import imageUploadSlice from "./slices/imageUploadSlice";

export default configureStore({
    reducer: {
        imageUpload: imageUploadSlice
    },
    middleware: (getDefaultMiddleware) => getDefaultMiddleware({
        serializableCheck: false
    })
})