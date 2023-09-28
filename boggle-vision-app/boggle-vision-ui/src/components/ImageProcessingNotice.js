// This component is the ImageProcessingNotice, which will be displayed when the user uploads an image. 

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

import React from 'react';
import { useSelector } from 'react-redux/es/hooks/useSelector';
import axios from 'axios';
import { useState, useEffect } from 'react';

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the ImageProcessingNotice component.

const ImageProcessingNotice = () => {

    // This state will contain the response from the server.
    const [response, setResponse] = useState(null);

    // The imageUploadSlice will contain the image that the user uploaded.
    const imageUploadSlice = useSelector(state => state.imageUpload);

    // This useEffect will be called when the imageUploadSlice.image changes.
    useEffect(()=> {
        // If the image is not null, then send the image to the server.
        if (imageUploadSlice.image !== null) {
            const endpointURL = "http://127.0.0.1:8000/analyze_image";
            axios.post(endpointURL, {
                image: imageUploadSlice.image
            }).then((response) => {
                console.log(response)
                setResponse(response.data);
            }).catch((error) => {
                setResponse(JSON.stringify(error));
            })
        }
    }, [imageUploadSlice.image])

    // If the image is null, then return null.
    if (imageUploadSlice.image === null) {
        return null;
    }

    else {
        // Otherwise, return the following JSX.
        return (
            <div>
                {JSON.stringify(response)}
            </div>
        )
    }

    // else if (response === null) {
    //     const endpointURL = "http://127.0.0.1:8000/";
    //     axios.get(endpointURL).then((response) => {
    //         setResponse(response.data);
    //     }).catch((error) => {
    //         setResponse(JSON.stringify(error));
    //     })
    // }

    // Otherwise, return the following JSX.
    return (
        <div>
            {JSON.stringify(response)}
        </div>
    )
}

// Export the ImageProcessingNotice component.
export default ImageProcessingNotice;