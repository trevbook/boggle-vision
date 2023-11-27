import { Grid } from "@mantine/core";
import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import "./BoardStats.css";

const BoardStats = () => {
  const boardData = useSelector((state) => state.boardData);
  const [boardStats, setBoardStats] = useState(null);
  const [totalPoints, setTotalPoints] = useState(null);
  const [elevenPointWords, setElevenPointWords] = useState(null);
  const [wordCount, setWordCount] = useState(null);
  const [longestWord, setLongestWord] = useState(null);
  const [avgPointsPerWord, setAvgPointsPerWord] = useState(null);

  useEffect(() => {
    setBoardStats(boardData.boardStats);
  }, [JSON.stringify(boardData)]);

  useEffect(() => {
    if (boardStats) {
      setTotalPoints(boardStats.total_points);
      setElevenPointWords(boardStats.eleven_point_words);
      setWordCount(boardStats.word_count);
      setLongestWord({
        word: boardStats.longest_word,
        length: boardStats.longest_word_length,
      });
      setAvgPointsPerWord(boardStats.avg_points_per_word);
    }
  }, [boardStats]);

  const getZScoreText = (zScore) => {
    if (zScore > 0.5) {
      return "Above average! 🚀";
    } else if (zScore < -0.5) {
      return "Below average 😅";
    } else {
      return "Average 👍";
    }
  };

  // This function will return the JSX for a metric.
  const Metric = (props) => {
    return (
      <div className="board-stats-metric-container">
        <div
          className="board-stats-metric-header"
          style={{ ...props.headerStyleOverride }}
        >
          {props.header}
        </div>
        <div
          className="board-stats-metric-value"
          style={{ ...{ color: props.color }, ...props.valueStyleOverride }}
        >
          {props.value}
        </div>
        <div
          className="board-stats-metric-explanation"
          style={{ ...props.explanationStyleOverride }}
        >
          {props.explanationText}
        </div>
      </div>
    );
  };

  return (
    <div
      style={{
        width: "100%",
        wordWrap: "break-word",
        textAlign: "left",
      }}
    >
      {boardStats ? (
        <>
          <div style={{ marginBottom: "00px" }}>
            <Grid>
              <Grid.Col span={6}>
                <Metric
                  header="Total Points"
                  value={totalPoints}
                  color={boardStats.total_points_color}
                  //   explanationText={getZScoreText(boardStats.total_points_z_score)}
                />
              </Grid.Col>
              <Grid.Col span={6}>
                <Metric
                  header="Word Count"
                  value={wordCount}
                  color={boardStats.word_count_color}
                  //   explanationText={getZScoreText(boardStats.word_count_z_score)}
                />
              </Grid.Col>
              <Grid.Col span={6}>
                <Metric
                  header="11-Point Words"
                  value={elevenPointWords}
                  color={boardStats.eleven_pointers_color}
                  //   explanationText={getZScoreText(
                  //     boardStats.eleven_point_words_z_score
                  //   )}
                />
              </Grid.Col>

              <Grid.Col span={6}>
                <Metric
                  header="PPW"
                  value={avgPointsPerWord}
                  color={boardStats.avg_points_per_word_color}
                  //   explanationText={getZScoreText(
                  //     boardStats.eleven_point_words_z_score
                  //   )}
                />
                {/* {longestWord ? (
                  <Metric
                    header="Longest Word"
                    value={longestWord.word}
                    color={boardStats.longest_word_color}
                    valueStyleOverride={{ fontSize: "2.5rem", fontWeight: 400 }}
                    explanationText={`${longestWord.length} letters`}
                    explanationStyleOverride={{ fontStyle: "italic" }}
                  />
                ) : (
                  <div></div>
                )} */}
              </Grid.Col>
            </Grid>
          </div>
        </>
      ) : (
        <p>Loading stats...</p>
      )}
    </div>
  );
};

export default BoardStats;
