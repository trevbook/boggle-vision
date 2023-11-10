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
import "./WordTable.css";

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
    var columnNames = ["word", "length", "points"];
    return columnNames.map((columnName) => {
      return {
        accessorKey: columnName,
        header: columnName.charAt(0).toUpperCase() + columnName.slice(1),
      };
    });
  }, [props.wordsTableData]);

  const table = useMantineReactTable({
    columns,
    data: props.wordsTableData,
    state: {
      pageSize: 40,
    },
    mantinePaperProps: {
      shadow: "none",
      sx: {
        borderRadius: "0",
        border: "none",
      },
      "elevation": 0
    },
    mantineTableHeadCellProps: {
      align: "center",
      sx: {
        paddingLeft: "0px"
      }
    },
    mantineTableHeadRowProps: {
      sx: {
        borderBottom: "1px solid orange",
      },
    },
    mantineTableProps: {
      sx: {
        border: "none"
      },
    },
    mantinePaperProps: {},
    mantineTableBodyRowProps: ({ row }) => ({
      // Define a click event that handles a user clicking on a row.
      onClick: (event) => {
        // Gently scroll to the top of the page.
        window.scrollTo({ top: -10, behavior: "smooth" });

        // Dispatch the action that'll set the selected word index.
        const selected_word_id = row.original.word_id;
        dispatch(setSelectedWordIndex(selected_word_id));
      },
      sx: {
        "&:hover": {
          backgroundColor: "#e0e0e0",
          cursor: "pointer",
        },
        border: "none",
      },
    }),
    initialState: { density: "xs", pageSize: 40 },
    style: { width: "100%" },
    sx: {
      tableLayout: "fixed",
    },
  });

  // Otherwise, we're going to render the WordTable.
  return (
    <div style={{ width: "100%" }}>
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
