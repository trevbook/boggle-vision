// This component is WordTable, which will contain a Mantine React Table
// with all of the words from the solved Boggle board.

// ==============================================================
//                              SETUP
// ==============================================================
// The following are imports of modules and components that are required to make this component work.

// Import statements for this file
import React, { useRef, useState, useEffect, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import axios from "axios";
import { MantineReactTable, useMantineReactTable } from "mantine-react-table";
import { setSelectedWordIndex } from "../../slices/userControlSlice";

// ==============================================================
//                        COMPONENT DEFINITION
// ==============================================================
// Below, we define the component.

/**
 * A component that contains the WordTable.
 */
const WordTable = (props) => {
  // Set up a dispatch.
  const dispatch = useDispatch();

  // We're going to keep the columns in this memoized variable.
  const columns = useMemo(() => {
    // If the wordsTableData is null, return an empty array.
    if (props.wordsTableData === null) {
      return null;
    }

    // Otherwise, we'll return each of the columns. Look at the
    // first word in the wordsTableData to get the columns.
    const firstWord = props.wordsTableData[0];
    const columnNames = Object.keys(firstWord);
    return columnNames.map((columnName) => {
      return {
        accessorKey: columnName,
        header: columnName,
      };
    });
  }, [props.wordsTableData]);

  const table = useMantineReactTable({
    columns,
    data: props.wordsTableData,
    mantineTableBodyRowProps: ({ row }) => ({
      // Define a click event that handles a user clicking on a row.
      onClick: (event) => {
        // Gently scroll to the top of the page.
        window.scrollTo({ top: -10, behavior: "smooth" });
        
        // Dispatch the action that'll set the selected word index.
        const selected_word_id = row.original.word_id 
        dispatch(setSelectedWordIndex(selected_word_id));

      },
    }),
    initialState: { density: "xs" },
    style: { width: "100%" },
    sx: {
      tableLayout: "fixed",
    },
  });

  // Otherwise, we're going to render the WordTable.
  return (
    <div style={{ width: "100%", border: "2px solid green" }}>
      {columns === null || props.wordsTableData === null ? (
        <div />
      ) : (
        <MantineReactTable table={table} />
      )}
    </div>
  );
};

// Export the component.
export default WordTable;
