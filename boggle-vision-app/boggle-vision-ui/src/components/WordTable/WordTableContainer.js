// This component is WordTableContainer, which will contain all of the 
// components necessary for the WordTable.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from 'react-redux';
import axios from "axios";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component. 

/**
 * A component that contains the WordTableContainer.
 */
const WordTableContainer = (
    props
) => {

    // Set up a selector for the board data.
    const boardDataSlice = useSelector(state => state.boardData);

    // Set up a state to store the solved board data.
    const [solvedBoardData, setSolvedBoardData] = useState(null);

    // Set up an effect to handle the changing of the board data.
    useEffect(() => {

        // If the boardData is null, then set the solvedBoardData to null.
        if (boardDataSlice.boardData === null) {
            setSolvedBoardData(null);
        }
        else {
            console.log("boardData:")
            console.log(boardDataSlice.boardData)
            // Otherwise, we're going to ping the solve_board endpoint.
            const endpointURL = window.location.hostname === 'localhost' ?
                "http://127.0.0.1:8000/solve_board" :
                "http://192.168.1.159:8000/solve_board";

            // Send the board data to the server.
            axios.post(endpointURL, boardDataSlice.boardData.letter_sequence).then((response) => {

                // Set the solved board data.
                setSolvedBoardData(response.data);
            }).catch(
                (error) => {
                    console.log("Error in WordTableContainer: " + JSON.stringify(error));
                    setSolvedBoardData(null);
                }
            ).finally(() => { });
        }



    }, [JSON.stringify(boardDataSlice.boardData)])

    // If props.wordTableData is null, then return null.
    if (solvedBoardData === null) {
        return (
            <div style={{ "width": "100%", "border": "2px solid green" }}>
                WordTableContainer
            </div>
        )
    }

    // Otherwise, we're going to render the WordTableContainer.
    return (
        <div style={{ "width": "100%", "border": "2px solid green" }}>
            {JSON.stringify(solvedBoardData)}
        </div>
    )

}

// Export the component.
export default WordTableContainer
