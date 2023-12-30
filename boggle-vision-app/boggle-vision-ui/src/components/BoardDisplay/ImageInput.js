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
import FilterOverlay from "./FilterOverlay";
import { Icon } from "@iconify/react";

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
  const fileInputRef = useRef(null); // Ref for the hidden file input
  const boardCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const wordOverlayCanvasRef = useRef(null);
  const filterOverlayCanvasRef = useRef(null);

  // Declare a dispatch
  const dispatch = useDispatch();

  // The imageUploadSlice will contain the image that the user uploaded.
  const imageUploadSlice = useSelector((state) => state.imageUpload);

  // The boardImages selector will contain the uploaded board data
  const boardImagesSlice = useSelector((state) => state.boardImages);

  // Height of the control buttons
  var controlContainerHeight = "30px";

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
        filterOverlayCanvasRef.current.width = img.width;
        filterOverlayCanvasRef.current.height = img.height;
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

  const uploadImage = () => {
    // Function to trigger the file input click
    fileInputRef.current.click();
  };

  const loadRandomImage = () => {
    // Generate a random number between 1 and 5
    const randomNum = Math.floor(Math.random() * 5) + 1;

    // Construct the image file path
    const imagePath = `/images/sample-image-${randomNum}.png`;

    // Create and load the image
    const image = new Image();
    image.crossOrigin = "Anonymous"; // Needed for CORS when converting to base64
    image.src = imagePath;
    image.onload = () => {
      // Create a canvas element
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;

      // Draw the image onto the canvas
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);

      // Remove alpha channel by drawing the image onto itself with a white background
      ctx.globalCompositeOperation = "destination-over";
      ctx.fillStyle = "#fff"; // White background
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Convert the canvas to a data URL (base64 string)
      const base64Image = canvas.toDataURL("image/jpeg"); // Convert to JPEG to discard alpha channel

      // Dispatch the action to set the image in your Redux store
      dispatch(setImage(base64Image));
    };
  };

  // Render the component.
  return (
    <div>
      <input
        type="file"
        id="hiddenFileInput"
        onChange={handleImageChange}
        ref={fileInputRef}
        style={{ display: "none" }} // Hide the file input
      />

      {/* This allows the upload of an image */}
      <label>
        {image ? (
          // Make this div a flexbox that centers its contents.
          <div className="capture-container">
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
                  zIndex: 3,
                  maxWidth: "100%",
                  height: "100%",
                }}
              />

              {/* Canvas for the word overlay */}
              <canvas
                ref={wordOverlayCanvasRef}
                style={{
                  position: "absolute",
                  zIndex: 4,
                  maxWidth: "100%",
                  height: "100%",
                }}
              />

              {/* Canvas for the word overlay */}
              <canvas
                ref={filterOverlayCanvasRef}
                style={{
                  position: "absolute",
                  zIndex: 2,
                  maxWidth: "100%",
                  height: "100%",
                }}
              />

              {/* Your TileOverlay component */}
              <FilterOverlay filterOverlayCanvasRef={filterOverlayCanvasRef} />
              <TileOverlay overlayCanvasRef={overlayCanvasRef} />
              <WordOverlay wordOverlayCanvasRef={wordOverlayCanvasRef} />
            </div>
          </div>
        ) : (
          <div>
            <div className="uploadButtonContainer">
              <div className="iconContainer">
                <Icon
                  icon={"uil:camera"}
                  color={"black"}
                  onClick={() => {
                    uploadImage();
                  }}
                  height={controlContainerHeight}
                />
                <div className={"`iconDescription"}>Upload Image</div>
              </div>

              {/* Random Image Button */}
              <div className="iconContainer">
                <Icon
                  icon={"fa-solid:dice"}
                  color={"black"}
                  onClick={() => {
                    loadRandomImage();
                  }}
                  height={controlContainerHeight}
                />
                <div className={"`iconDescription"}>Random Image</div>
              </div>
            </div>
            <ImageProcessingNotice />
          </div>
        )}
      </label>
    </div>
  );
};

// Export the component.
export default ImageInput;
