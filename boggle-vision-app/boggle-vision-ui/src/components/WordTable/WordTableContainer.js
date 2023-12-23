// This component is WordTableContainer, which will contain all of the
// components necessary for the WordTable.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import axios from "axios";
import { setWordsTableData, setBoardStats } from "../../slices/boardDataSlice";
import WordTable from "./WordTable";
import { setWordToLetterContourPath } from "../../slices/boardImagesSlice";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the WordTableContainer.
 */
const WordTableContainer = (props) => {
  // Set up a selector for the board data.
  const boardDataSlice = useSelector((state) => state.boardData);

  // Set up a selector for the solved board data.
  const solvedBoardData = useSelector(
    (state) => state.boardData.wordsTableData
  );

  // Set up a dispatch.
  const dispatch = useDispatch();

  // Set up an effect to handle the changing of the board data.
  useEffect(() => {
    // If the boardData is null, then set the solvedBoardData to null.
    if (boardDataSlice.boardData === null) {
      // Dispatch the action that'll set the word table data.
      dispatch(setWordsTableData(null));
      dispatch(setBoardStats(null));
    } else {
      // // Otherwise, we're going to ping the solve_board endpoint.
      // const endpointURL =
      //   window.location.hostname === "localhost"
      //     ? "http://127.0.0.1:8000/solve_board"
      //     : "http://192.168.1.159:8000/solve_board";

      const apiBaseUrl =
        window.location.hostname === "localhost"
          ? "http://127.0.0.1:8000"
          : "http://34.171.53.77:9781";
      const endpointURL = `${apiBaseUrl}/solve_board`;

      // Send the board data to the server.
      axios
        .post(endpointURL, boardDataSlice.boardData.letter_sequence)
        .then((response) => {
          // Unpack the data from the response
          const words_table_data = response.data.solved_board;
          const board_stats = response.data.board_stats;
          const word_id_to_path = response.data.word_id_to_path;

          // Set the solved board data.
          dispatch(setWordsTableData(words_table_data));
          dispatch(setBoardStats(board_stats));
          dispatch(setWordToLetterContourPath(word_id_to_path));
        })
        .catch((error) => {
          dispatch(setWordsTableData(null));
          dispatch(setBoardStats(null));
        })
        .finally(() => {});
    }
  }, [JSON.stringify(boardDataSlice.boardData)]);

  // If props.wordTableData is null, then return null.
  if (solvedBoardData === null) {
    return <div style={{ width: "100%" }}>WordTableContainer</div>;
  }

  // Otherwise, we're going to render the WordTableContainer.
  return (
    <div style={{ width: "100%" }}>
      <WordTable wordsTableData={solvedBoardData} />
    </div>
  );
};

// Export the component.
export default WordTableContainer;
