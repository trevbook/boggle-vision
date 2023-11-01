// This is the TileOverlay component. It will enable some highlights to
// be drawn "over" the board image.

// Below, you'll find the import statements for this file.
import { Grid } from "@mantine/core";
import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";

// Now: we're going to define the TileOverlay component!
const TileOverlay = (props) => {
  // Extract the overlayCanvasRef from the props.
  const { overlayCanvasRef } = props;

  // Set up a selector for the boardDataSlice.
  const boardDataSlice = useSelector((state) => state.boardData);

  // Set up a selector for the userControlSlice.
  const userControlSlice = useSelector((state) => state.userControl);

  // Set up a selector for the boardImagesSlice.
  const boardImagesSlice = useSelector((state) => state.boardImages);

  // Set up an effect to draw the tile overlay.
  useEffect(() => {
    // If the overlayCanvasRef is null, leave this effect.
    if (!overlayCanvasRef) {
      return;
    }

    // If the userControlSlice.show_letter_overlay is false, leave this effect.
    if (!userControlSlice.show_letter_overlay) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      setTimeout(() => {
        ctx.clearRect(
          0,
          0,
          overlayCanvasRef.current.width,
          overlayCanvasRef.current.height
        );
      }, 30);
      return;
    }

    // If the boardDataSlice.letter_sequence is null, leave this effect.
    if (!boardDataSlice.letterSequence) {
      return;
    }

    // If the boardImagesSlice.letterImageContours is null, leave this effect.
    if (!boardImagesSlice.letterImageContours) {
      return;
    }

    // If the boardImagesSlice.boardImageOriginalHeight is null, leave this effect.
    if (
      !boardImagesSlice.boardImageOriginalHeight ||
      !boardImagesSlice.boardImageOriginalWidth
    ) {
      return;
    }

    // If we've made it this far, we're going to draw the tile overlay. For each of the contours in
    // the boardImagesSlice.letterImageContours, we're going to draw a neon green square. At the top-left of
    // this green square, in VERY small font, we'll write the letter associated with the contour.
    // Get the canvas context
    const ctx = overlayCanvasRef.current.getContext("2d");

    // Delete everything on the canvas
    ctx.clearRect(
      0,
      0,
      overlayCanvasRef.current.width,
      overlayCanvasRef.current.height
    );

    // Loop through the keys in letterImageContours
    Object.keys(boardImagesSlice.letterImageContours).forEach((key) => {
      const contour = boardImagesSlice.letterImageContours[key];

      const imageWidthScale =
        overlayCanvasRef.current.height /
        boardImagesSlice.boardImageOriginalWidth;
      const imageHeightScale =
        overlayCanvasRef.current.height /
        boardImagesSlice.boardImageOriginalHeight;

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
      ctx.strokeStyle = "#13fc03";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
      ctx.fill();

      // Draw the corresponding letter
      const avgX = scaledContour.reduce((acc, point) => acc + point[0], 0) / 4;
      const avgY = scaledContour.reduce((acc, point) => acc + point[1], 0) / 4;
      var letter = boardDataSlice.letterSequence[key];

      // If the letter is a block, make it a square
      if (letter == "BLOCK") {
        letter = "■";
      }
      const textHeight = 45; // Since the font size is set to 60px
      ctx.font = `bold ${textHeight}px Arial`;

      ctx.fillStyle = "#13fc03";
      // Calculate the width and height of the text
      const textWidth = ctx.measureText(letter).width;

      // Adjust avgX and avgY to consider the center point
      const centeredX = avgX - textWidth / 2;
      const centeredY = avgY + textHeight / 2 - 15; // Adjust the Y offset by considering the baseline of the font

      ctx.fillText(letter, centeredX, centeredY);
    });
  }, [userControlSlice.show_letter_overlay]);

  return <div></div>;
};

export default TileOverlay;
