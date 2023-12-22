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
import {
  setSelectedWordData,
  setSelectedWordIndex,
  setSelectedWordRarityColor,
} from "../../slices/userControlSlice";
import axios from "axios";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

const rarity_to_hex_color = (rarity) => {
  if (rarity === null) {
    return "#000000";
  }

  // Otherwise, we're going to return the appropriate color.
  switch (rarity) {
    case "Common":
      return "#138004";
    case "Uncommon":
      return "#005ab1";
    case "Rare":
      return "#862dc2";
    case "Very Rare":
      return "#cc6600";
    default:
      return "#000000";
  }
};

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

  // Set up a dispatch variable.
  const dispatch = useDispatch();

  // This effect will set the selected word.
  useEffect(() => {
    // If any of the dependencies are null, then return.
    if (selected_word_index === null || wordsTableData === null) {
      return;
    }

    // Otherwise, we're going to set the selected word.
    setSelectedWord(wordsTableData[selected_word_index].word);
  }, [selected_word_index, wordsTableData]);

  // Set up an effect to get the word stats from the API
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

    // // Otherwise, we're going to ping the word_rarity endpoint.
    // const endpointURL =
    //   window.location.hostname === "localhost"
    //     ? "http://127.0.0.1:8000/word_rarity"
    //     : "http://192.168.1.159:8000/word_rarity";

    const apiBaseUrl = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
      const endpointURL = `${apiBaseUrl}/word_rarity`;

    // Send the word to the server.
    axios
      .post(endpointURL, { word: word })
      .then((response) => {
        // Unpack the data from the response.

        // If the response is null, then return.
        if (response.data === null) {
          return;
        }

        // Otherwise, we're going to set the word stats.
        setWordStats(response.data);
        dispatch(setSelectedWordData(response.data));
        dispatch(
          setSelectedWordRarityColor(rarity_to_hex_color(response.data.rarity))
        );
      })

      .catch((error) => {
        setWordStats("error fetching stats");
      })
      .finally(() => {});
  }, [selected_word_index, wordsTableData]);

  return (
    // If the wordStats is null, then return null.
    wordStats === null ? null : (
      <div style={{ width: "100%" }}>
        <Grid>
          <Col span={6} style={{ paddingTop: "4px" }}>
            {/* Below, we'll have the wordStats.length, and a boldened label under it saying "Length".
          Everything will be centered. */}
            <div style={{ textAlign: "center" }}>
              <div className={"word-stat-label"}>Length</div>
              <div className={"word-stat-value"}>{wordStats.length}</div>
            </div>
          </Col>
          <Col span={6} style={{ paddingTop: "4px" }}>
            {/* Now do the same, but for points */}
            <div style={{ textAlign: "center" }}>
              <div className={"word-stat-label"}>Points</div>
              <div className={"word-stat-value"}>{wordStats.points}</div>
            </div>
          </Col>
          <Col span={12} style={{ paddingBottom: "7px", paddingTop: "7px" }}>
            <div
              className={"word-rarity-div"}
              style={{
                textAlign: "center",
                color: rarity_to_hex_color(wordStats.rarity),
              }}
            >
              {wordStats.rarity}
            </div>
          </Col>
        </Grid>
      </div>
    )
  );
};

// Export the component.
export default SelectedWordInfoStats;
