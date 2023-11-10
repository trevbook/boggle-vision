// This is the FilterOverlay component. It will enable visual filters to be
// drawn over the board image.

// Below, you'll find the import statements for this file.
import { Grid } from "@mantine/core";
import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";

// Now: we're going to define the FilterOverlay component!
const FilterOverlay = (props) => {
  // Extract the filterOverlayCanvasRef from the props.
  const { filterOverlayCanvasRef } = props;

  // Set up some of the selectors that this component will use
  const current_visual_filter = useSelector(
    (state) => state.userControl.current_visual_filter
  );
  const letterImageContours = useSelector(
    (state) => state.boardImages.letterImageContours
  );
  const boardImageOriginalWidth = useSelector(
    (state) => state.boardImages.boardImageOriginalWidth
  );
  const boardImageOriginalHeight = useSelector(
    (state) => state.boardImages.boardImageOriginalHeight
  );
  const letterImageActivations = useSelector(
    (state) => state.boardImages.letterImageActivations
  );
  const boardImages = useSelector((state) => state.boardImages.boardImages);

  // We're going to keep a state that holds an image object for the activation heatmap.
  const [activation_heatmap_image, set_activation_heatmap_image] =
    useState(null);

  useEffect(() => {
    if (boardImages == null) {
      return;
    }

    const activation_heatmap = boardImages.activation_heatmap;
    const new_activation_heatmap_image = new Image();

    // Set the source which will trigger the onload event
    new_activation_heatmap_image.src = `data:image/png;base64,${activation_heatmap}`;

    new_activation_heatmap_image.onload = () => {
      set_activation_heatmap_image(new_activation_heatmap_image);
    };
  }, [JSON.stringify(boardImages)]);

  useEffect(() => {
    // If the current_visual_filter is null, we're going to
    // remove any existing image from the canvas.
    if (!current_visual_filter && filterOverlayCanvasRef && activation_heatmap_image) {
      const overlayCanvas = filterOverlayCanvasRef.current;
      const ctx = overlayCanvas.getContext("2d");

      // Delete everything on the canvas
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); 

    }

    if (activation_heatmap_image == null || filterOverlayCanvasRef == null) {
      return;
    }

    if (current_visual_filter === "feature_activations") {
      // Moved the canvas and context inside the load event handler
      const overlayCanvas = filterOverlayCanvasRef.current;
      const ctx = overlayCanvas.getContext("2d");

      // Set the canvas size if necessary - uncomment if you want to match image size
      overlayCanvas.width = activation_heatmap_image.width;
      overlayCanvas.height = activation_heatmap_image.height;

      // Now we can draw the image because it is guaranteed to have been loaded
      ctx.drawImage(activation_heatmap_image, 0, 0);
      console.log("drew the image");
    }
  }, [
    activation_heatmap_image,
    current_visual_filter
    // You may remove selectors from the dependency array if they are not used in this effect
  ]);

  return <div></div>;
};

export default FilterOverlay;
