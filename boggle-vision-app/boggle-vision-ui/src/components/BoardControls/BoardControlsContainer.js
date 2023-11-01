// This component is the BoardControlsContainer, which will contain the controls
// for Boggle Vision.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Icon } from "@iconify/react";
import { Grid, MantineProvider, Modal } from "@mantine/core";
import { setBoardImages } from "../../slices/boardImagesSlice";
import { setBoardData, setBoardStats } from "../../slices/boardDataSlice";
import { setImage } from "../../slices/imageUploadSlice";
import { useDisclosure } from "@mantine/hooks";
import EditModalContent from "./EditModal/EditModalContent";
import {
  setSelectedWordIndex,
  toggleLetterOverlay,
} from "../../slices/userControlSlice";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

const controlContainerHeight = "30px";

const styles = {
  container: {
    width: "100%",
    display: "flex",
    justifyContent: "space-between",
    textAlign: "center",
    paddingBottom: "10px",
    height: 75,
    backgroundColor: "white",
  },
  iconContainer: {
    flex: "1",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    height: 75,
  },
  iconDescription: {
    marginTop: "5px",
    fontSize: "0.9rem",
    fontWeight: 600,
  },
};

const BoardControlsContainer = () => {
  const dispatch = useDispatch();
  const [opened, { open, close }] = useDisclosure();

  // Set up some selectors for this component
  const show_letter_overlay = useSelector((state) => state.userControl.show_letter_overlay);
  const selected_word_index = useSelector((state) => state.userControl.selected_word_index);


  return (
    <div>
      <Modal opened={opened} onClose={close} title="Edit Board">
        <EditModalContent />
      </Modal>
      <div style={styles.container}>
        {[
          {
            icon: "mdi:image-edit-outline",
            color: "green",
            description: "Edit Image",
          },
          {
            icon: "mdi:eraser",
            onClick: () => {
              dispatch(setSelectedWordIndex(null));
            },
            color: selected_word_index === null ? "grey" : "pink",
            description: "Erase",
          },
          {
            icon: show_letter_overlay
              ? "mdi:alphabetical"
              : "mdi:alphabetical-off",
            color: show_letter_overlay ? "orange" : "grey",
            onClick: () => {
              dispatch(toggleLetterOverlay(true));
            },
            description: show_letter_overlay
              ? "Letters"
              : "Letters",
          },
          {
            icon: "mdi:rotate-left-variant",
            color: "blue",
            description: "Rotate Left",
          },
          {
            icon: "mdi:pencil-outline",
            color: "black",
            onClick: open,
            description: "Edit Board",
          },
          {
            icon: "mdi:close-thick",
            color: "red",
            onClick: () => {
              dispatch(setBoardImages(null));
              dispatch(setBoardData(null));
              dispatch(setImage(null));
              dispatch(setBoardStats(null));
            },
            description: "Clear Board",
          },
        ].map(({ icon, color, onClick, description }, idx) => (
          <div key={idx} style={styles.iconContainer}>
            <Icon
              icon={icon}
              color={color}
              onClick={onClick}
              height={controlContainerHeight}
            />
            <div style={styles.iconDescription}>{description}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BoardControlsContainer;
