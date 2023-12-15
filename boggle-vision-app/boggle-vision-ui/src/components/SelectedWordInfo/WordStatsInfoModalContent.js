import React, { useState, useEffect } from "react";
import { useSelector } from "react-redux";

const WordStatsInfoModalContent = () => {
  // Set up a selector for the boardDataSlice.
  // TODO: This is a pretty hacky way of doing it. I should just have 
  // the selected word data in the userControlSlice.
  const wordsTableData = useSelector((state) => state.boardData.wordsTableData);
  const selected_word_index = useSelector(
    (state) => state.userControl.selected_word_index
  );

  const selected_word_data = useSelector(
    (state) => state.userControl.selected_word_data
  );

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

  if (!selected_word_data) {
    return null;
  }

  const { ct, length, points, pct_games, z_score, rarity, total_games } =
    selected_word_data;
  const percentage = (pct_games * 100).toFixed(4); // Convert to percentage string with 4 decimal places

  // You may need to create this utility function to format numbers with commas.
  function numberWithCommas(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  return (
    <div
      style={{
        padding: "20px",
        backgroundColor: "#f4f4f4",
        borderRadius: "8px",
        color: "#333",
      }}
    >
      <h2 style={{ color: "#444" }}>Word Statistics</h2>
      <h3 style={{ color: "#444", marginBottom: "10px" }}>{selectedWord}</h3>
      <p>
        I simulated {numberWithCommas(total_games)} games, and this word came up{" "}
        {numberWithCommas(ct)} times.
      </p>
      <p>
        <strong>Length:</strong> {length} 
      </p>
      <p>
        <strong>Points:</strong> {points}
      </p>
      <p>
        <strong>Frequency:</strong> {percentage}% of games
      </p>
      <p>
        <strong>Z-Score:</strong> {z_score.toFixed(3)} std. dev's from the average word frequency
      </p>
      <p>
        <strong>Rarity:</strong> {rarity}
      </p>
    </div>
  );
};

export default WordStatsInfoModalContent;
