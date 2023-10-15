// This component is BoardStats, which will contain some statistics 
// about the Boggle board. 

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useRef } from "react";
import { useDispatch, useSelector } from 'react-redux';
import boardDataSlice from "../../slices/boardDataSlice";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component. 

/**
 * A component that contains the BoardStats.
 */
const BoardStats = (
) => {

    // Set up a selector for the board data.
    const boardData = useSelector(state => state.boardData);

    // Render the component.
    return (
        <div style={{"width": "100%", "border": "2px solid orange"}}>
            {boardData.boardData !== null ? <div style={{
                "width": "100%", "wordBreak": "break-word"
            }}>
                {JSON.stringify(boardData.boardData)}
            </div> : "No board data"}
        </div>
    )

}

// Export the component.
export default BoardStats
