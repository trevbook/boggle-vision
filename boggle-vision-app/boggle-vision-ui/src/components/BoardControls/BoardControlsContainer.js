// This component is the BoardControlsContainer, which will contain the controls
// for Boggle Vision. 

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useRef } from "react";
import { useDispatch, useSelector } from 'react-redux';
import { Icon } from '@iconify/react'
import { Grid, MantineProvider, Modal } from "@mantine/core";
import { setImage } from "../../slices/imageUploadSlice";
import { setBoardData } from "../../slices/boardDataSlice";
import { useDisclosure } from "@mantine/hooks";
import EditModalContent from "./EditModal/EditModalContent";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component. 

/**
 * A component that contains the BoardControls.
 */
const BoardControlsContainer = (
) => {

    // Declare a dispatch
    const dispatch = useDispatch();

    // Declare the useDisclosure hook.
    const [opened, { open, close }] = useDisclosure();

    // Styles for the parent container
    const containerStyle = {
        width: '100%',
        border: '2px solid blue',
        display: 'flex',   // Use flexbox
        justifyContent: 'space-between',  // Distribute items evenly
        textAlign: 'center',
        "paddingTop": "10px",
        "paddingBottom": "10px"
    };

    // Styles for each icon container
    const iconContainerStyle = {
        flex: '1',  // Each child takes up an equal amount of space
        display: 'flex',
        justifyContent: 'center',  // Center icon horizontally
        alignItems: 'center'  // Center icon vertically
    };

    // Render the component.
    return (
        <div>
            <Modal opened={opened} onClose={close} title="Edit Board">
                <EditModalContent />
            </Modal>
            <div style={containerStyle}>
                <div style={iconContainerStyle}>
                    <Icon icon="mdi:image-edit-outline" color="green" />
                </div>
                <div style={iconContainerStyle}>
                    <Icon icon="mdi:eraser" color="pink" />
                </div>
                <div style={iconContainerStyle}>

                    <Icon icon="mdi:layers-outline" color="orange" />
                </div>
                <div style={iconContainerStyle}>
                    <Icon icon="mdi:rotate-left-variant" color="blue" />

                </div>

                {/* BUTTON #5:  */}
                {/* EDIT BOARD  */}
                <div style={iconContainerStyle}>
                    <Icon icon="mdi:pencil-outline" color="black" onClick={open} />
                </div>


                {/* BUTTON #6:   */}
                {/* CLEAR BOARD  */}
                <div style={iconContainerStyle}>
                    <Icon icon="mdi:close-thick" color="red" onClick={() => {
                        console.log("Clearing image")
                        dispatch(setImage(null))
                        // Dispatch the action that'll set the board data.
                        dispatch(setBoardData(null));
                    }} />
                </div>
            </div>
        </div>

    )

}

// Export the component.
export default BoardControlsContainer
