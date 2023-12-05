// This component is SelectedWordInfoPanel, which will be a container for information
// about the currently selected word.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import {
  Card,
  CloseButton,
  Col,
  Grid,
  Text,
  UnstyledButton,
} from "@mantine/core";
import React, { useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import SelectedWordInfoContent from "./SelectedWordInfoContent";
import "./SelectedWordInfoPanel.css";
import SelectedWordInfoControls from "./SelectedWordInfoControls";
import { setSelectedWordIndex } from "../../slices/userControlSlice";
import SelectedWordInfoStats from "./SelectedWordInfoStats";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the SelectedWordInfoPanel.
 */
const SelectedWordInfoPanel = () => {
  const selected_word_index = useSelector(
    (state) => state.userControl.selected_word_index
  );
  const panelClass =
    selected_word_index !== null ? "panelVisible" : "panelHidden";

  // This constant will define the height of the panel.
  const PANEL_HEIGHT = 175;

  // Declare a dispatch variable.
  const dispatch = useDispatch();

  return (
    <div className={`selectedWordInfoPanel ${panelClass}`}>
      <Card style={{ height: PANEL_HEIGHT }}>
        <Grid style={{ height: PANEL_HEIGHT }}>
          <Col span={6} style={{ paddingRight: "10px" }}>
            <SelectedWordInfoContent />
          </Col>
          <Col span={6} style={{ height: "100%" }}>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                height: "100%",
              }}
            >
              <div
                style={{
                  flex: 0.4,
                  justifyContent: "right",
                  alignItems: "start",
                  display: "flex",
                }}
              >
                <CloseButton
                  onClick={() => {
                    dispatch(setSelectedWordIndex(null));
                  }}
                />
              </div>
              <div
                style={{
                  flex: 1,

                  justifyContent: "center",
                  alignItems: "center",
                  display: "flex",
                  border: "1px solid blue",
                }}
              >
                <SelectedWordInfoStats />
              </div>
              <div
                style={{
                  flex: 0.1,
                  justifyContent: "center",
                  alignItems: "center",
                  paddingBottom: "5px",
                }}
              >
                <SelectedWordInfoControls />
              </div>
            </div>
          </Col>
        </Grid>
      </Card>
    </div>
  );
};

// Export the component.
export default SelectedWordInfoPanel;
