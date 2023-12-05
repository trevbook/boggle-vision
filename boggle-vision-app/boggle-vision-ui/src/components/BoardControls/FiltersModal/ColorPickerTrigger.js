/**
 * FileName.js
 *
 * This is a description of what the file/module does and any other relevant information
 * that you would like to include, such as authorship, date of creation, purpose of the
 * file, dependencies, etc. This block of comment can be read by developers and can be
 * made to show up in IntelliSense or other code editors like VSCode.
 *
 * @module FileName
 * @author Your Name
 * @date Date Created
 * @dependency {Module/Package Name}
 */

// Below, you'll find the import statements for this component
import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import "./ColorPickerTrigger.css";
import { ColorPicker, Popover } from "@mantine/core";

const ColorPickerTrigger = ({ label, color, onChange }) => {
  // Set up a State to control whether or not this particular ColorPicker
  // is actually opened
  const [opened, setOpened] = useState(false);

  // This function will be called when the Popover needs to be closed.
  // It can also be used in other places if needed to programmatically close the Popover.
  const closePopover = () => setOpened(false);

  // This is the return statement, which controls what renders for the component
  return (
    <div
      style={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      <Popover
        position="bottom"
        withArrow
        opened={opened}
        onClose={closePopover}
        trapFocus={false}
        closeOnEscape={true}
        transition="pop-top-left"
        closeOnClickOutside={false}

      >
        <Popover.Target>
          <button
            style={{
              background: "none",
              border: "none",
              padding: 0,
              margin: 0,
              cursor: "pointer",
            }}
            onClick={() => setOpened((o) => !o)}
          >
            <div
              style={{
                width: "30px",
                height: "30px",
                borderRadius: "50%",
                background: color,
                cursor: "pointer",
                border: "2px solid black",
              }}
              onClick={(event) => {
                //   event.stopPropagation();
                // setOpened((o) => !o);
              }}
            ></div>
          </button>
        </Popover.Target>
        <div className="color-picker-label">{label}</div>
        <Popover.Dropdown
          className={opened ? "popover-centered" : ""}
          onClick={(event) => {
            event.stopPropagation();
          }}
        >
          <div className="color-picker-container">
            <ColorPicker
              onChangeEnd={(color) => {
                onChange(color);
                setOpened(false);
              }}
              value={color}
              format={"hex"}
              fullWidth
            />
          </div>
        </Popover.Dropdown>
      </Popover>
    </div>
  );
};

// Export the ColorPickerTrigger component for use in other components
export default ColorPickerTrigger;
