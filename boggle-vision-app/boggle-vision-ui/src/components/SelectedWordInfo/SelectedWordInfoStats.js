// This is the SelectedWordInfoStats component.
// It contains the stats for the currently selected word in the SelectedWordInfoPanel.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import { Card, Col, Grid, Text, UnstyledButton } from "@mantine/core";
import React, { useRef, useState, useEffect } from "react";
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
 * A component that contains the SelectedWordInfoStats.
 */
const SelectedWordInfoStats = () => {
  // Set up a selector for the boardDataSlice.
  const wordsTableData = useSelector((state) => state.boardData.wordsTableData);
  const selected_word_index = useSelector(
    (state) => state.userControl.selected_word_index
  );

  // Otherwise, we're going to return the selected word.
  // const selected_word = wordsTableData[selected_word_index].word;

  // Set up a state for the word stats.
  const [wordStats, setWordStats] = useState(null);

  // Set up a state for the selected word
  const [selectedWord, setSelectedWord] = useState(null);

  // This effect will set the selected word.
  useEffect(() => {
    // If any of the dependencies are null, then return.
    if (selected_word_index === null || wordsTableData === null) {
      return;
    }

    // Otherwise, we're going to set the selected word.
    setSelectedWord(wordsTableData[selected_word_index].word);
  }, [selected_word_index, wordsTableData]);

  // This effect will get the word stats from the API.
  // EXAMPLE CODE FROM SelectedWordInfoContent:
  //    // Set up an effect to get the word definition from the API
  //    useEffect(() => {
  //     // If the selected_word_index is null, or the wordsTableData is null, then return.
  //     if (selected_word_index === null || wordsTableData === null) {
  //       return;
  //     }

  //     // Try and get the word from the wordsTableData.
  //     const word = wordsTableData[selected_word_index].word;

  //     // If the word is null, then return.
  //     if (word === null) {
  //       return;
  //     }

  //     // Otherwise, we're going to ping the define_word endpoint.
  //     const endpointURL =
  //       window.location.hostname === "localhost"
  //         ? "http://127.0.0.1:8000/define_word"
  //         : "http://192.168.1.159:8000/define_word";

  //     // Send the word to the server.
  //     axios
  //       .post(endpointURL, { word: word })
  //       .then((response) => {
  //         // Unpack the data from the response.
  //         if (response.data === null) {
  //           setWordDefinition("no definition found");
  //         } else {
  //           setWordDefinition(response.data.definition);
  //         }
  //       })
  //       .catch((error) => {
  //         setWordDefinition("error fetching definition");
  //       })
  //       .finally(() => {});
  //   }, [selected_word_index, wordsTableData]);

  return <div>This is where the stats will go.</div>;
};

// Export the component.
export default SelectedWordInfoStats;
