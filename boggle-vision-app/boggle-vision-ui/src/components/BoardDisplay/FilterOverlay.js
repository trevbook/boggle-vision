// This is the FilterOverlay component. It will enable visual filters to be
// drawn over the board image.

// Below, you'll find the import statements for this file.
import { Grid } from "@mantine/core";
import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { hexToRgb, interpolateColor } from "../../utils/image_manipulation";

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

  // Set up a selector to grab the feature_activations_filter_primary_color and
  // feature_activations_filter_secondary_color from the userControlSlice
  const featureActivationsFilterPrimaryColor = useSelector(
    (state) => state.userControl.feature_activations_filter_primary_color
  );
  const featureActivationsFilterSecondaryColor = useSelector(
    (state) => state.userControl.feature_activations_filter_secondary_color
  );

  // Set up another selector to grab the canny_edge_filter_primary_color and 
  // canny_edge_filter_secondary_color from the userControlSlice
  const cannyEdgeFilterPrimaryColor = useSelector(
    (state) => state.userControl.canny_edge_filter_primary_color
  );
  const cannyEdgeFilterSecondaryColor = useSelector(
    (state) => state.userControl.canny_edge_filter_secondary_color
  );

  // We're going to keep a state that holds an image object for the activation heatmap.
  const [activation_heatmap_image, set_activation_heatmap_image] =
    useState(null);
  
  // We'll also keep a state that holds an image object for the Canny Edge Detection heatmap
  const [canny_edge_image, set_canny_edge_image] = useState(null)

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

    // Do the same thing for the canny edge image
    const canny_edge = boardImages.canny_edge_viz;
    const new_canny_edge_image = new Image();
    new_canny_edge_image.src = `data:image/png;base64,${canny_edge}`
    new_canny_edge_image.onload = () => {
      set_canny_edge_image(new_canny_edge_image)
    }

  }, [JSON.stringify(boardImages)]);

  useEffect(() => {
    // If the current_visual_filter is null, we're going to
    // remove any existing image from the canvas.
    if (
      !current_visual_filter &&
      filterOverlayCanvasRef &&
      activation_heatmap_image
    ) {
      const overlayCanvas = filterOverlayCanvasRef.current;
      const ctx = overlayCanvas.getContext("2d");

      // Delete everything on the canvas
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }

    if (activation_heatmap_image == null || filterOverlayCanvasRef == null || canny_edge_image == null) {
      return;
    }

    if (current_visual_filter === "feature_activations") {

      // Moved the canvas and context inside the load event handler
      const overlayCanvas = filterOverlayCanvasRef.current;
      const ctx = overlayCanvas.getContext("2d");
      ctx.drawImage(activation_heatmap_image, 0, 0);
      const imageData = ctx.getImageData(0, 0, filterOverlayCanvasRef.current.width, filterOverlayCanvasRef.current.height);
      const data = imageData.data;

      // Determine the primary and secondary colors
      var primaryRgb = "rgb(255, 255, 255)";
      if (featureActivationsFilterPrimaryColor) {
        primaryRgb = hexToRgb(featureActivationsFilterPrimaryColor);
      }
      var secondaryRgb = "rgb(0, 0, 0)";
      if (featureActivationsFilterSecondaryColor) {
        secondaryRgb = hexToRgb(featureActivationsFilterSecondaryColor);
      }

      for (let i = 0; i < data.length; i += 4) {
        // Convert the grayscale value to a factor between 0 and 1
        const factor = data[i] / 255;
        const [r, g, b] = interpolateColor(secondaryRgb, primaryRgb, factor);
        data[i] = r; // Red
        data[i + 1] = g; // Green
        data[i + 2] = b; // Blue
      }

      // Set the canvas size if necessary - uncomment if you want to match image size
      overlayCanvas.width = activation_heatmap_image.width;
      overlayCanvas.height = activation_heatmap_image.height;

      // Now we can draw the image because it is guaranteed to have been loaded
      
      // Put the image data back after manipulation
      ctx.putImageData(imageData, 0, 0);
      
    }


    // Deal with the scenario where the current_visual_filter is "canny_edge"
    if (current_visual_filter === "canny_edge") {
      // Moved the canvas and context inside the load event handler
      const overlayCanvas = filterOverlayCanvasRef.current;
      const ctx = overlayCanvas.getContext("2d");
      ctx.drawImage(canny_edge_image, 0, 0);
      const imageData = ctx.getImageData(0, 0, filterOverlayCanvasRef.current.width, filterOverlayCanvasRef.current.height);
      const data = imageData.data;

      // Determine the primary and secondary colors
      var primaryRgb = "rgb(255, 255, 255)";
      if (cannyEdgeFilterPrimaryColor) {
        primaryRgb = hexToRgb(cannyEdgeFilterPrimaryColor);
      }
      var secondaryRgb = "rgb(0, 0, 0)";
      if (cannyEdgeFilterSecondaryColor) {
        secondaryRgb = hexToRgb(cannyEdgeFilterSecondaryColor);
      }

      for (let i = 0; i < data.length; i += 4) {
        // Convert the grayscale value to a factor between 0 and 1
        const factor = data[i] / 255;
        const [r, g, b] = interpolateColor(secondaryRgb, primaryRgb, factor);
        data[i] = r; // Red
        data[i + 1] = g; // Green
        data[i + 2] = b; // Blue
      }

      // Set the canvas size if necessary - uncomment if you want to match image size
      overlayCanvas.width = canny_edge_image.width;
      overlayCanvas.height = canny_edge_image.height;

      // Now we can draw the image because it is guaranteed to have been loaded
      
      // Put the image data back after manipulation
      ctx.putImageData(imageData, 0, 0);
    }


  }, [
    activation_heatmap_image,
    canny_edge_image,
    current_visual_filter,
    featureActivationsFilterPrimaryColor,
    featureActivationsFilterSecondaryColor,
    cannyEdgeFilterPrimaryColor,
    cannyEdgeFilterSecondaryColor
  ]);

  return <div></div>;
};

export default FilterOverlay;
