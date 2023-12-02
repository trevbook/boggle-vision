// This component is SelectedWordInfoPanel, which will be a container for information
// about the currently selected word.

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
import SelectedWordInfoControls from "./SelectedWordInfoControls";

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

  return (
    <div className={`selectedWordInfoPanel ${panelClass}`}>
      <Card style={{ height: 175 }}>
        <Grid style={{ height: "100%" }}>
          <Col span={6} style={{ paddingRight: "10px" }}>
            <SelectedWordInfoContent />
          </Col>
          <Col span={6}>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                height: "100%",
              }}
            >
              <div
                style={{
                  flex: 1,

                  justifyContent: "center",
                  alignItems: "center",
                  display: "flex",
                }}
              >
                Stats go here
              </div>
              <div
                style={{
                  flex: 1,
                  justifyContent: "center",
                  alignItems: "center",
                  display: "flex",
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
