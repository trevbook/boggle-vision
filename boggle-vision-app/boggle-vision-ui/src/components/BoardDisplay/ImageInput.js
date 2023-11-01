// This component is the ImageInput, which will allow the user to upload an image.
// It will also display the image that the user has uploaded!

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useEffect, useRef } from "react";
import "./ImageInput.css";
import { useDispatch } from "react-redux";
import { useSelector } from "react-redux/es/hooks/useSelector";
import { setImage } from "../../slices/imageUploadSlice";
import ImageProcessingNotice from "./ImageProcessingNotice";
import TileOverlay from "./TileOverlay";
import WordOverlay from "./WordOverlay";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the ImageInput component.

/**
 * A component that allows the user to take a picture of their Boggle board.
 */
const ImageInput = () => {
  const [image, setImageState] = React.useState(null);

  // Declare a ref for the "overlay canvas"
  const boardCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const wordOverlayCanvasRef = useRef(null);

  // Declare a dispatch
  const dispatch = useDispatch();

  // The imageUploadSlice will contain the image that the user uploaded.
  const imageUploadSlice = useSelector((state) => state.imageUpload);

  // The boardImages selector will contain the uploaded board data
  const boardImagesSlice = useSelector((state) => state.boardImages);

  // This useEffect will be called when the imageUploadSlice.image changes.
  useEffect(() => {
    if (
      boardImagesSlice.boardImages !== null &&
      boardImagesSlice.boardImages.cropped_board !== null
    ) {
      setImageState(
        `data:image/png;base64,${boardImagesSlice.boardImages.cropped_board}`
      );
    }

    // Check if the image is null.
    else if (imageUploadSlice.image !== null) {
      // Set the image state.
      setImageState(imageUploadSlice.image);
    }

    // Otherwise, set the image state to null.
    else {
      setImageState(null);
    }
  }, [boardImagesSlice.boardImages, imageUploadSlice.image]);

  // This effect will be called when the user enables the letter overlay
  useEffect(() => {
    if (image) {
      const img = new Image();
      img.src = image;
      img.onload = () => {
        const boardCanvas = boardCanvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;
        const ctx = boardCanvas.getContext("2d");
        boardCanvas.width = img.width;
        boardCanvas.height = img.height;
        overlayCanvas.width = img.width;
        overlayCanvas.height = img.height;
        wordOverlayCanvasRef.current.width = img.width;
        wordOverlayCanvasRef.current.height = img.height;
        ctx.drawImage(img, 0, 0);
      };
    }
  }, [image]);

  const handleImageChange = (e) => {
    // This function is called when the user uploads an image.

    // Get the file.
    const file = e.target.files[0];
    const reader = new FileReader();

    // Once the file is read, set the image state.
    reader.onload = () => {
      dispatch(setImage(reader.result));
    };

    // Read the file.
    if (file) {
      reader.readAsDataURL(file);
    }
  };

  // Render the component.
  return (
    <div>
      {/* This allows the upload of an image */}
      <label htmlFor="hiddenFileInput" className="capture-container">
        <input type="file" id="hiddenFileInput" onChange={handleImageChange} />
        {image ? (
          // Make this div a flexbox that centers its contents.
          <div style={{ display: "flex", justifyContent: "center" }}>
            {/* Canvas for the board */}
            <canvas
              ref={boardCanvasRef}
              style={{
                position: "absolute",
                zIndex: 1,
                maxWidth: "100%",
                height: "100%",
              }}
            ></canvas>

            {/* Canvas for the overlay */}
            <canvas
              ref={overlayCanvasRef}
              style={{
                position: "absolute",
                zIndex: 2,
                maxWidth: "100%",
                height: "100%",
              }}
            />

            {/* Canvas for the word overlay */}
            <canvas
              ref={wordOverlayCanvasRef}
              style={{
                position: "absolute",
                zIndex: 3,
                maxWidth: "100%",
                height: "100%",
              }}
            />

            {/* Your TileOverlay component */}
            <TileOverlay overlayCanvasRef={overlayCanvasRef} />

            <WordOverlay wordOverlayCanvasRef={wordOverlayCanvasRef} />
          </div>
        ) : (
          <div className="custom-file-button">
            <ImageProcessingNotice />
          </div>
        )}
      </label>
    </div>
  );
};

// Export the component.
export default ImageInput;
