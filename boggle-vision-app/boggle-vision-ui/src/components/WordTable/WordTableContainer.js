// This component is WordTableContainer, which will contain all of the 
// components necessary for the WordTable.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useRef } from "react";
import { useDispatch, useSelector } from 'react-redux';

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component. 

/**
 * A component that contains the WordTableContainer.
 */
const WordTableContainer = (
) => {

    // Render the component.
    return (
        <div style={{"width": "100%", "border": "2px solid green"}}>
            WordTableContainer
        </div>
    )

}

// Export the component.
export default WordTableContainer
