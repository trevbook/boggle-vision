// Import statements for this file
import React, { useRef } from "react";
import "./ImageInputComponent.css";

/**
 * A component that allows the user to take a picture of their Boggle board.
 */
const ImageInputComponent = (
) => {

    // Declare references to the video and canvas elements.
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    // Initialize the camera.
    const startCamera = async () => {
        const video = videoRef.current;
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }
        })
        video.srcObject = stream;
    }

    // Capture the image from a video stream, and draw it on a canvas.
    const captureImage = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    // Render the component.
    return (
        <div className="capture-container">

            {/* This allows the upload of an image */}
            <input type="file" accept="image/*"></input>

            {/* This allows the capture of an image from a video stream */}
            <div className="video-wrapper">
                <video ref={videoRef} autoPlay />
                <div className="frame-overlay"></div>
            </div>

            <button onClick={startCamera}>Start Camera</button>
            <button onClick={captureImage}>Capture Image</button>

            {/* This allows the display of an image */}
            <canvas ref={canvasRef} width="640" height="480" />
        </div>
    )

}

export default ImageInputComponent