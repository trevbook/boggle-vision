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
import "./SelectedWordInfoPanel.css"

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the SelectedWordInfoPanel.
 */
const SelectedWordInfoPanel = () => {
  const selected_word_index = useSelector((state) => state.userControl.selected_word_index);
  const panelClass = selected_word_index !== null ? 'panelVisible' : 'panelHidden';

  return (
    <div className={`selectedWordInfoPanel ${panelClass}`}>
      <Card shadow="sm" radius="md" style={{ height: '100px', "border": "2px solid blue", 
    "textAlign": "left"}}>
        <SelectedWordInfoContent />
      </Card>
    </div>
  );
};

// Export the component.
export default SelectedWordInfoPanel;
