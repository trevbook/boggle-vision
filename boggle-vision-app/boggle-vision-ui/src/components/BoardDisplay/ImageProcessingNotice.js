// This component is the ImageProcessingNotice, which will be displayed when the user uploads an image.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

import React from "react";
import { useSelector } from "react-redux/es/hooks/useSelector";
import axios from "axios";
import { useState, useEffect } from "react";
import { useDispatch } from "react-redux";
import { setBoardData } from "../../slices/boardDataSlice";
import {
  setBoardImages,
  setLetterImageActivations,
} from "../../slices/boardImagesSlice";
import { setLetterImageContours } from "../../slices/boardImagesSlice";
import { setLetterSequence } from "../../slices/boardDataSlice";
import { setBoardImageOriginalHeight } from "../../slices/boardImagesSlice";
import { setBoardImageOriginalWidth } from "../../slices/boardImagesSlice";
import { setLoading } from "../../slices/imageUploadSlice";
import { Icon } from "@iconify/react";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the ImageProcessingNotice component.

const ImageProcessingNotice = () => {
  // This state will contain the response from the server.
  const [response, setResponse] = useState(null);

  // Declare a dispatch
  const dispatch = useDispatch();

  // The imageUploadSlice will contain the image that the user uploaded.
  const imageUploadSlice = useSelector((state) => state.imageUpload);

  // Declare some styling for the icons.
  var container = {
    width: "100%",
    display: "flex",
    justifyContent: "space-between",
    textAlign: "center",
    paddingBottom: "10px",
    height: 75,
    backgroundColor: "white",
  };
  var iconContainer = {
    flex: "1",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    height: 75,
  };
  var iconDescription = {
    marginTop: "5px",
    fontSize: "0.85rem",
    fontWeight: 600,
  };
  var controlContainerHeight = "30px";

  // This useEffect will be called when the imageUploadSlice.image changes.
  useEffect(() => {
    // If the image is not null, then send the image to the server.
    if (imageUploadSlice.image !== null) {
      const apiBaseUrl =
        window.location.hostname === "localhost"
          ? "http://127.0.0.1:8000"
          : "http://34.171.53.77:9781";
      const endpointURL = `${apiBaseUrl}/analyze_image`;

      // Set the loading state to True
      dispatch(setLoading(true));

      // Send the image to the server.
      axios
        .post(endpointURL, {
          image: imageUploadSlice.image,
        })
        .then((response) => {
          // We'll start by unpacking the boardData
          const letter_sequence = response.data.letter_sequence;
          const cropped_board_image_str = response.data.cropped_board;
          const tile_contours = response.data.tile_contours;
          const cropped_board_width = response.data.cropped_board_width;
          const cropped_board_height = response.data.cropped_board_height;
          const letter_activation_visualization_list =
            response.data.letter_activation_visualization_list;
          const activation_heatmap = response.data.activation_heatmap;
          const canny_edge_viz = response.data.canny_edge_visualization;

          // Make a boardData object
          const cur_board_data = {
            letter_sequence: letter_sequence,
            tile_contours: tile_contours,
          };

          // Dispatch the action that'll set the board data.
          dispatch(setBoardData(cur_board_data));

          // Dispatch the action that'll set the board images.
          dispatch(
            setBoardImages({
              cropped_board: cropped_board_image_str,
              activation_heatmap: activation_heatmap,
              canny_edge_viz: canny_edge_viz,
            })
          );

          // Dispatch the action that'll set the letter image contours.
          dispatch(setLetterImageContours(tile_contours));

          // Dispatch the action that'll set the letter sequence.
          dispatch(setLetterSequence(letter_sequence));

          // Dispatch some of the height and width data
          dispatch(setBoardImageOriginalHeight(cropped_board_height));
          dispatch(setBoardImageOriginalWidth(cropped_board_width));

          // Dispatch the setting of the letter_activation_visualization_list
          dispatch(
            setLetterImageActivations(letter_activation_visualization_list)
          );
        })
        .catch((error) => {
          const error_display_str = "Error: " + JSON.stringify(error);
          setResponse(error_display_str);
        })
        .finally(() => {
          // Wait a bit, to give the user a chance to see the loading message.
          setTimeout(() => {
            // Reset loading to false when the API call is finished.
            dispatch(setLoading(false));
          }, 1000);
        });
    } else {
      // Reset the response to null.
      setResponse(null);
    }
  }, [imageUploadSlice.image]);

  return (
    <div>
      {response === null ? (
        <div />
      ) : (
        <div style={{ width: "100%", wordWrap: "break-word" }}></div>
      )}
    </div>
  );
};

// Export the ImageProcessingNotice component.
export default ImageProcessingNotice;
