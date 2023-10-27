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

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

const controlContainerHeight = "30px"

const styles = {
  container: {
    width: "100%",
    display: "flex",
    justifyContent: "space-between",
    textAlign: "center",
    paddingTop: "10px",
    paddingBottom: "10px",
    height: "100px",
    backgroundColor: "white",
    boxShadow: "0px 0px 10px 0px rgba(0,0,0,0.75)",
  },
  iconContainer: {
    flex: "1",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    height: 100
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
          { icon: "mdi:eraser", color: "pink", description: "Erase" },
          {
            icon: "mdi:layers-outline",
            color: "orange",
            description: "Layers",
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
            <Icon icon={icon} color={color} onClick={onClick} height={controlContainerHeight} />
            <div style={styles.iconDescription}>{description}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BoardControlsContainer;
