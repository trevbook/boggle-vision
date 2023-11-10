// This component is FiltersModalContent, which will contain the content for a modal
// that allows a user to select the current filter.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import { Card, Col, Grid, Text, UnstyledButton } from "@mantine/core";
import React, { useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { setCurrentVisualFilter } from "../../../slices/userControlSlice";

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

  // Use the special `children` prop to implicitly pass the content.
  const CardButton = ({ onClick, children }) => (
    <UnstyledButton onClick={onClick}>
      {/* Make the card take up 100% of the width */}
      <Card shadow="sm" style={{ width: "100%" }}>
        {children} {/* Use the children prop here */}
      </Card>
    </UnstyledButton>
  );

  return (
    <div id="overarching">
      <Grid>
        {/* NO FILTER */}
        <Col span={12}>
          <CardButton
            onClick={() => {
              dispatch(setCurrentVisualFilter(null));
            }}
          >
            <Text fw={500}>No Filter</Text>
            <Text size="sm" c="dimmed">
              This is the original image of the Boggle board, cropped to the
              board itself. No filters are applied.
            </Text>
          </CardButton>
        </Col>

        {/* ACTIVATION HEATMAP FILTER */}
        <Col span={12}>
          <CardButton
            onClick={() => {
              dispatch(setCurrentVisualFilter("feature_activations"));
            }}
          >
            <Text fw={500}>Feature Activations</Text>
            <Text size="sm" c="dimmed">
              This is a heatmap of the feature activations of the board image.
            </Text>
          </CardButton>
        </Col>
      </Grid>
    </div>
  );
};

// Export the component.
export default FiltersModalContent;
