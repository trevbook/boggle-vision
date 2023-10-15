// This component is the ImageInput, which will allow the user to upload an image. 
// It will also display the image that the user has uploaded!

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useEffect, useRef } from "react";
import "./ImageInput.css";
import { useDispatch } from 'react-redux';
import { useSelector } from 'react-redux/es/hooks/useSelector';
import { setImage } from "../../slices/imageUploadSlice";
import ImageProcessingNotice from "./ImageProcessingNotice";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the ImageInput component.



/**
 * A component that allows the user to take a picture of their Boggle board.
 */
const ImageInput = (
) => {

    const [image, setImageState] = React.useState(null)

    // Declare a dispatch
    const dispatch = useDispatch();

    // The imageUploadSlice will contain the image that the user uploaded.
    const imageUploadSlice = useSelector(state => state.imageUpload);

    // This useEffect will be called when the imageUploadSlice.image changes.
    useEffect(() => {

        // Check if the image is null.
        if (imageUploadSlice.image !== null) {
            // Set the image state.
            setImageState(imageUploadSlice.image)
        }

        // Otherwise, set the image state to null.
        else {
            setImageState(null)
        }
    }
        , [imageUploadSlice.image])

    const handleImageChange = (e) => {
        // This function is called when the user uploads an image.

        // Get the file.
        const file = e.target.files[0];
        const reader = new FileReader();

        // Once the file is read, set the image state.
        reader.onload = () => {
            dispatch(setImage(reader.result))
        }

        // Read the file.
        if (file) {
            reader.readAsDataURL(file);
        }

    }

    // Render the component.
    return (
        <div >
            {/* This allows the upload of an image */}
            <label htmlFor="hiddenFileInput" className="capture-container">
                <input
                    type="file"
                    id="hiddenFileInput"
                    onChange={handleImageChange}
                />
                {
                    image ? (
                        <img style={{ "maxWidth": "100%", }} src={image} alt="Image uploaded"></img>
                    ) : (
                        <div className="custom-file-button">
                            <ImageProcessingNotice />
                        </div>
                    )
                }
            </label>

        </div>
    )

}

// Export the component.
export default ImageInput
