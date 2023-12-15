// This component is FiltersModalContent, which will contain the content for a modal
// that allows a user to select the current filter.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import { Card, Col, Grid, Text, UnstyledButton } from "@mantine/core";
import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { setCurrentVisualFilter } from "../../../slices/userControlSlice";
import ColorPickerTrigger from "./ColorPickerTrigger";
import {
  setFeatureActivationsFilterPrimaryColor,
  setFeatureActivationsFilterSecondaryColor,
  setCannyEdgeFilterPrimaryColor,
  setCannyEdgeFilterSecondaryColor,
} from "../../../slices/userControlSlice";
import { interpolateColor, hexToRgb } from "../../../utils/image_manipulation";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the FiltersModalContent.
 */
const FiltersModalContent = () => {
  // Set up the dispatch
  const dispatch = useDispatch();

  // Set up some selectors for the feature_activations_filter_primary_color and
  // feature_activations_filter_secondary_color from the userControlSlice.
  const featureActivationsFilterPrimaryColor = useSelector(
    (state) => state.userControl.feature_activations_filter_primary_color
  );
  const featureActivationsFilterSecondaryColor = useSelector(
    (state) => state.userControl.feature_activations_filter_secondary_color
  );

  // Grab these same states for the Canny Edge filter
  const cannyEdgeFilterPrimaryColor = useSelector(
    (state) => state.userControl.canny_edge_filter_primary_color
  );
  const cannyEdgeFilterSecondaryColor = useSelector(
    (state) => state.userControl.canny_edge_filter_secondary_color
  );

  // Set up a selector that grabs the current_visual_filter from the userControlSlice.
  const currentVisualFilter = useSelector(
    (state) => state.userControl.current_visual_filter
  );

  // This state will hold the modified image source
  const [modifiedImageSrc, setModifiedImageSrc] = useState(
    "/images/feature-visualization-example.jpg"
  );

  // This state will hold the modified image source for the Canny Edge Detection filter
  const [canny_edge_modified_image_src, set_canny_edge_modified_image_src] =
    useState("/images/canny.png");

  // This effect will modify the image source
  useEffect(() => {
    // Load the original image
    const image = new Image();
    image.src = "/images/feature-visualization-example.jpg";
    image.onload = () => {
      // Create an offscreen canvas
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext("2d");

      // Draw the original image onto the canvas
      ctx.drawImage(image, 0, 0);

      // Get the image data from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
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

      ctx.putImageData(imageData, 0, 0);

      // After manipulation, convert the canvas to a Data URL
      const dataUrl = canvas.toDataURL();

      // Update the state with the new image source
      setModifiedImageSrc(dataUrl);
    };
  }, [
    featureActivationsFilterPrimaryColor,
    featureActivationsFilterSecondaryColor,
  ]);

  // Now, this will be the effect for the Canny Edge Detection filter
  useEffect(() => {
    // Load the original image
    const image = new Image();
    image.src = "/images/canny.png";
    image.onload = () => {
      // Create an offscreen canvas
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext("2d");

      // Draw the original image onto the canvas
      ctx.drawImage(image, 0, 0);

      // Get the image data from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
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

      ctx.putImageData(imageData, 0, 0);

      // After manipulation, convert the canvas to a Data URL
      const dataUrl = canvas.toDataURL();

      // Update the state with the new image source
      set_canny_edge_modified_image_src(dataUrl);
    };
  }, [cannyEdgeFilterPrimaryColor, cannyEdgeFilterSecondaryColor]);

  // Declare a styling fo rthe cards
  const cardStyle = {
    height: "100%", // Make sure the card takes full height of the column
    border: "1px solid black",
    shadow: "sm",
  };

  // Use the special `children` prop to implicitly pass the content.
  const CardButton = ({ onClick, children, active }) => {
    // Declare a styling for the card that's active
    const activeCardStyle = {
      ...cardStyle,
      border: active ? "3px solid black" : "1px solid black",
    };

    return (
      <UnstyledButton
        onClick={onClick}
        style={{ width: "100%", height: "100%" }}
      >
        {/* Make the card take up 100% of the width */}
        <Card shadow="sm" style={activeCardStyle}>
          {children} {/* Use the children prop here */}
        </Card>
      </UnstyledButton>
    );
  };

  return (
    <div id="overarching">
      <Grid style={{ alignItems: "stretch" }}>
        {/* NO FILTER */}
        <Col span={12}>
          <CardButton
            onClick={() => {
              dispatch(setCurrentVisualFilter(null));
            }}
            active={currentVisualFilter === null}
          >
            <img
              src="/images/no-filter-example.jpg"
              alt="A Boggle board with no filter applied."
              style={{
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
                width: "50%",
                marginBottom: "10px",
              }}
            />
            <Text fw={500}>No Filter</Text>
            <Text size="xs" c="dimmed">
              This is the original image of the Boggle board. No filters will be
              applied.
            </Text>
          </CardButton>
        </Col>

        {/* ACTIVATION HEATMAP FILTER */}
        <Col span={12}>
          <CardButton
            onClick={() => {
              dispatch(setCurrentVisualFilter("feature_activations"));
            }}
            active={currentVisualFilter === "feature_activations"}
          >
            <img
              src={modifiedImageSrc}
              alt="A Boggle board with no filter applied."
              style={{
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
                width: "50%",
                marginBottom: "10px",
              }}
            />
            <div style={{ marginBottom: 20 }}>
              <Text fw={500}>Feature Activations</Text>
              <Text size="xs" c="dimmed">
                A heatmap of the activations of the convolutional filters.
              </Text>
            </div>
            <div>
              <Grid>
                <Col span={6}>
                  <ColorPickerTrigger
                    label={"Primary"}
                    color={featureActivationsFilterPrimaryColor}
                    onChange={(color) => {
                      dispatch(setFeatureActivationsFilterPrimaryColor(color));
                    }}
                  ></ColorPickerTrigger>
                </Col>
                <Col span={6}>
                  <ColorPickerTrigger
                    label={"Secondary"}
                    color={featureActivationsFilterSecondaryColor}
                    onChange={(color) => {
                      dispatch(
                        setFeatureActivationsFilterSecondaryColor(color)
                      );
                    }}
                  ></ColorPickerTrigger>
                </Col>
              </Grid>
            </div>
          </CardButton>
        </Col>

        {/* CANNY EDGE DETECTION FILTER */}
        <Col span={12}>
          <CardButton
            onClick={() => {
              dispatch(setCurrentVisualFilter("canny_edge"));
            }}
            active={currentVisualFilter === "canny_edge"}
          >
            <img
              src={canny_edge_modified_image_src}
              alt="A Boggle board with a canny edge detection filter applied"
              style={{
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
                width: "50%",
                marginBottom: "10px",
              }}
            />
            <div style={{ marginBottom: 20 }}>
              <Text fw={500}>Canny Edge Detection</Text>
              <Text size="xs" c="dimmed">
                This filter shows the Canny edge detection for a board.
              </Text>
            </div>
            <div>
              <Grid>
                <Col span={6}>
                  <ColorPickerTrigger
                    label={"Primary"}
                    color={cannyEdgeFilterPrimaryColor}
                    onChange={(color) => {
                      dispatch(setCannyEdgeFilterPrimaryColor(color));
                    }}
                  ></ColorPickerTrigger>
                </Col>
                <Col span={6}>
                  <ColorPickerTrigger
                    label={"Secondary"}
                    color={cannyEdgeFilterSecondaryColor}
                    onChange={(color) => {
                      dispatch(setCannyEdgeFilterSecondaryColor(color));
                    }}
                  ></ColorPickerTrigger>
                </Col>
              </Grid>
            </div>
          </CardButton>
        </Col>
      </Grid>
    </div>
  );
};

// Export the component.
export default FiltersModalContent;
