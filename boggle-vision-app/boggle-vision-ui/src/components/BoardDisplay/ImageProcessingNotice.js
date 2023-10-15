// This component is the ImageProcessingNotice, which will be displayed when the user uploads an image. 

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

import React from 'react';
import { useSelector } from 'react-redux/es/hooks/useSelector';
import axios from 'axios';
import { useState, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { setBoardData } from '../../slices/boardDataSlice';

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the ImageProcessingNotice component.

const ImageProcessingNotice = () => {

    // This state will contain the response from the server.
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);

    // Declare a dispatch
    const dispatch = useDispatch();

    // The imageUploadSlice will contain the image that the user uploaded.
    const imageUploadSlice = useSelector(state => state.imageUpload);

    // This useEffect will be called when the imageUploadSlice.image changes.
    useEffect(() => {
        // If the image is not null, then send the image to the server.
        if (imageUploadSlice.image !== null) {

            // Set the loading to true when the API call is made.
            setLoading(true);
            console.log("Setting loading to true")

            // Determine the endpoint URL.
            const endpointURL = window.location.hostname === 'localhost' ?
                "http://127.0.0.1:8000/analyze_image" :
                "http://192.168.1.159:8000/analyze_image";

            // Send the image to the server.
            axios.post(endpointURL, {
                image: imageUploadSlice.image
            }).then((response) => {

                // Dispatch the action that'll set the board data.
                dispatch(setBoardData(response.data));

            }).catch((error) => {
                const error_display_str = "Error: " + JSON.stringify(error);
                setResponse(error_display_str);
            }).finally(() => {

                // Wait a bit, to give the user a chance to see the loading message.
                setTimeout(() => {
                    // Reset loading to false when the API call is finished.
                    setLoading(false);
                }, 1000);
            })
        }

        else {
            // Reset the response to null.
            setResponse(null);
        }

    }, [imageUploadSlice.image])

    return (
        <div>
            {
                response === null ? "Click to upload an image..." : loading ? "Loading..." : <div style={{ "width": "100%", "wordWrap": "break-word" }}>

                </div>
            }
        </div>
    )
}

// Export the ImageProcessingNotice component.
export default ImageProcessingNotice;