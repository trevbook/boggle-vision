// This component is SelectedWordInfoContent, which will contain info about the
// currently selected word.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import { Card, Col, Grid, Text, UnstyledButton } from "@mantine/core";
import React, { useRef, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import axios from "axios";

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
  const selected_word_index = useSelector(
    (state) => state.userControl.selected_word_index
  );

  // Otherwise, we're going to return the selected word.
  // const selected_word = wordsTableData[selected_word_index].word;

  // Set up a state for the word definition.
  const [wordDefinition, setWordDefinition] = useState(null);

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

  // Set up an effect to get the word definition from the API
  useEffect(() => {
    // If the selected_word_index is null, or the wordsTableData is null, then return.
    if (selected_word_index === null || wordsTableData === null) {
      return;
    }

    // Try and get the word from the wordsTableData.
    const word = wordsTableData[selected_word_index].word;

    // If the word is null, then return.
    if (word === null) {
      return;
    }

    // AS FOLLOWS IS SOME EXAMPLE CODE:
    //     // Otherwise, we're going to ping the solve_board endpoint.
    //     const endpointURL =
    //     window.location.hostname === "localhost"
    //       ? "http://127.0.0.1:8000/solve_board"
    //       : "http://10.0.0.48:8000/solve_board";

    //   // Send the board data to the server.
    //   axios
    //     .post(endpointURL, boardDataSlice.boardData.letter_sequence)
    //     .then((response) => {
    //       // Unpack the data from the response
    //       const words_table_data = response.data.solved_board;
    //       const board_stats = response.data.board_stats;
    //       const word_id_to_path = response.data.word_id_to_path;

    //       // Set the solved board data.
    //       dispatch(setWordsTableData(words_table_data));
    //       dispatch(setBoardStats(board_stats));
    //       dispatch(setWordToLetterContourPath(word_id_to_path));
    //     })
    //     .catch((error) => {
    //       dispatch(setWordsTableData(null));
    //       dispatch(setBoardStats(null));
    //     })
    //     .finally(() => {});
    // }

    // Otherwise, we're going to ping the define_word endpoint.
    const endpointURL =
      window.location.hostname === "localhost"
        ? "http://127.0.0.1:8000/define_word"
        : "http://10.0.0.48:8000/define_word";

    // Send the word to the server.
    axios
      .post(endpointURL, { word: word })
      .then((response) => {
        // Unpack the data from the response.
        if (response.data === null) {
          setWordDefinition("no definition found");
        }
        else {
          setWordDefinition(response.data.definition);
        }
      })
      .catch((error) => {
        console.log(error);
        setWordDefinition("error fetching definition");
      })
      .finally(() => {});
  }, [selected_word_index, wordsTableData]);

  // If there is no selected word, return a message saying so.
  if (selected_word_index === null) {
    return <div></div>;
  }

  return (
    <div>
      <div>
        <Text>Selected Word: {selectedWord}</Text>
        <Text>Definition: {wordDefinition}</Text>
      </div>
    </div>
  );
};

// Export the component.
export default SelectedWordInfoContent;
