// This is the WordOverlay component. It will enable some highlights to
// be drawn "over" the board image for words that're present in the board.

// Below, you'll find the import statements for this file.
import { Grid } from "@mantine/core";
import React, { useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";

const ANIMATION_DELAY_PER_TILE = 130;

// Now: we're going to define the WordOverlay component!
const WordOverlay = (props) => {
  // Extract the wordOverlayCanvasRef from the props.
  const { wordOverlayCanvasRef } = props;

  // Set up some of the selectors for this component.
  const wordToLetterContourPath = useSelector(
    (state) => state.boardImages.wordToLetterContourPath
  );
  const letterImageContours = useSelector(
    (state) => state.boardImages.letterImageContours
  );
  const boardImageOriginalHeight = useSelector(
    (state) => state.boardImages.boardImageOriginalHeight
  );
  const boardImageOriginalWidth = useSelector(
    (state) => state.boardImages.boardImageOriginalWidth
  );
  const selected_word_index = useSelector(
    (state) => state.userControl.selected_word_index
  );
  const selected_word_rarity_color = useSelector(
    (state) => state.userControl.selected_word_rarity_color
  );

  // Helper function to draw a single tile
  const drawTile = (
    ctx,
    contour,
    imageWidthScale,
    imageHeightScale,
    color,
    lineWidth
  ) => {
    const scaledContour = contour.map((point) => [
      point[0] * imageWidthScale,
      point[1] * imageHeightScale,
    ]);

    ctx.beginPath();
    ctx.moveTo(scaledContour[0][0], scaledContour[0][1]);
    for (let i = 1; i < scaledContour.length; i++) {
      ctx.lineTo(scaledContour[i][0], scaledContour[i][1]);
    }
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  };

  // Declare a ref to store timeout IDs
  const timeoutIDs = useRef([]);

  // Set up an effect to draw the word overlay.
  useEffect(() => {
    // If the wordOverlayCanvasRef is null, leave this effect.
    if (!wordOverlayCanvasRef) {
      return;
    }

    // Wait 20ms before doing anything to make sure the rarity_colors are loaded.
    setTimeout(() => {}, 40);

    // Clear existing timeouts if any
    timeoutIDs.current.forEach(clearTimeout);
    timeoutIDs.current = [];

    // If the selected_word_index is null, leave this effect.
    if (selected_word_index === null) {
      // Clear everything on the canvas
      const ctx = wordOverlayCanvasRef.current.getContext("2d");
      ctx.clearRect(
        0,
        0,
        wordOverlayCanvasRef.current.width,
        wordOverlayCanvasRef.current.height
      );

      return;
    }

    // If the boardImagesSlice.letterImageContours is null, leave this effect.
    if (!letterImageContours) {
      return;
    }

    // If the boardImagesSlice.boardImageOriginalHeight is null, leave this effect.
    if (!boardImageOriginalHeight || !boardImageOriginalWidth) {
      return;
    }

    // If we've made it this far, we're going to draw the word overlay.
    const ctx = wordOverlayCanvasRef.current.getContext("2d");

    // Clear the canvas
    ctx.clearRect(
      0,
      0,
      wordOverlayCanvasRef.current.width,
      wordOverlayCanvasRef.current.height
    );

    // Calculate some scaling factors
    const imageWidthScale =
      wordOverlayCanvasRef.current.height / boardImageOriginalWidth;
    const imageHeightScale =
      wordOverlayCanvasRef.current.height / boardImageOriginalHeight;

    // Determine the contours we're going to draw
    const contour_idx_list = wordToLetterContourPath[selected_word_index];

    // Animate each tile with a delay
    contour_idx_list.forEach((contour_idx, index) => {
      const timeoutId = setTimeout(() => {
        const contour = letterImageContours[contour_idx];
        drawTile(
          ctx,
          contour,
          imageWidthScale,
          imageHeightScale,
          selected_word_rarity_color,
          8
        );
      }, index * ANIMATION_DELAY_PER_TILE); // 500ms delay between each tile

      // Store timeout ID
      timeoutIDs.current.push(timeoutId);
    });

    // Now, we'll loop through each of the key, value pairs in letterImageContours
    Object.keys(letterImageContours).forEach((contour_idx) => {
      // Check if the contour_idx is in contour_idx_list
      if (contour_idx_list.includes(parseInt(contour_idx))) {
        return;
      }

      // Get the contour
      const contour = letterImageContours[contour_idx];

      const scaledContour = contour.map((point) => [
        point[0] * imageWidthScale,
        point[1] * imageHeightScale,
      ]);

      // Start drawing
      ctx.beginPath();

      // Move to the first point
      ctx.moveTo(scaledContour[0][0], scaledContour[0][1]);

      // Draw lines to the other points
      for (let i = 1; i < scaledContour.length; i++) {
        ctx.lineTo(scaledContour[i][0], scaledContour[i][1]);
      }

      // Close the path and stroke, making sure that the background of each square is black
      ctx.closePath();
      ctx.strokeStyle = "rgb(235, 235, 235)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "rgb(0, 0, 0, 0.55)";
      ctx.fill();
    });
  }, [
    selected_word_index,
    wordToLetterContourPath,
    letterImageContours,
    boardImageOriginalHeight,
    boardImageOriginalWidth,
    wordOverlayCanvasRef,
    selected_word_rarity_color,
  ]);

  return <div></div>;
};

export default WordOverlay;
