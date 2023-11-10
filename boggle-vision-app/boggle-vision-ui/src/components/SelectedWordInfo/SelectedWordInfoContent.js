// This component is SelectedWordInfoContent, which will contain info about the
// currently selected word.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import { Card, Col, Grid, Text, UnstyledButton } from "@mantine/core";
import React, { useRef } from "react";
import { useDispatch, useSelector } from "react-redux";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the SelectedWordInfoContent.
 */
const SelectedWordInfoContent = () => {

  // Set up a selector for the boardDataSlice.
  const wordsTableData = useSelector((state) => state.boardData.wordsTableData);
  const selected_word_index = useSelector((state) => state.userControl.selected_word_index);

  // If there is no selected word, return a message saying so.
    if (selected_word_index === null) {
        return <div></div>;
    }

    // Otherwise, we're going to return the selected word.
    const selected_word = wordsTableData[selected_word_index].word;

  return <div>{`Selected word: ${selected_word}`}</div>;
};

// Export the component.
export default SelectedWordInfoContent;
