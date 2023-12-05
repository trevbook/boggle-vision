// This is the SelectedWordInfoControls component.
// It contains the controls for the SelectedWordInfoPanel.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import { Card, Col, Grid, Text, UnstyledButton } from "@mantine/core";
import React, { useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import SelectedWordInfoContent from "./SelectedWordInfoContent";
import "./SelectedWordInfoPanel.css";
import { Icon } from "@iconify/react";
import { setSelectedWordIndex } from "../../slices/userControlSlice";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the SelectedWordInfoControls.
 */
const SelectedWordInfoControls = () => {
  var iconContainer = {
    flex: "1",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
  };

  // Set up a dispatch variable.
  const dispatch = useDispatch();

  // Set up a selector for the selected_word_index variable.
  const selected_word_index = useSelector(
    (state) => state.userControl.selected_word_index
  );

  // Set up a Selector for the n_words variable.
  const n_words = useSelector((state) => state.boardData.n_words);

  return (
    <div>
      <Grid>
        {/* This column contains the Previous button. */}
        <Col span={6} style={{}}>
          <div style={iconContainer}>
            {/* When the selected_word_index is 0, we don't want to allow the user to go back; we'll need to "disable" the button */}
            <Icon
              icon={"ooui:previous-ltr"}
              color={"#000000"}
              onClick={() => {
                if (selected_word_index > 0) {
                  dispatch(setSelectedWordIndex(selected_word_index - 1));
                }
              }}
              height={"15px"}
              style={{ opacity: selected_word_index > 0 ? 1 : 0.5 }}
            />
            <Text style={{ fontSize: "0.7rem", fontWeight: 600 }}>Prev</Text>
          </div>
        </Col>

        {/* This column contains the Next button. */}
        <Col span={6}>
          <div style={iconContainer}>
            <Icon
              icon={"ooui:next-ltr"}
              color={"#000000"}
              onClick={() => {
                if (selected_word_index < n_words - 1) {
                  dispatch(setSelectedWordIndex(selected_word_index + 1));
                }
              }}
              height={"15px"}
              style={{ opacity: selected_word_index < n_words - 1 ? 1 : 0.5 }}
            />
            <Text style={{ fontSize: "0.7rem", fontWeight: 600 }}>Next</Text>
          </div>
        </Col>
      </Grid>
    </div>
  );
};

// Export the component.
export default SelectedWordInfoControls;
