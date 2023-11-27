import { Provider } from "react-redux";
import { MantineProvider, Grid, Modal } from "@mantine/core";
import ImageInput from "../components/BoardDisplay/ImageInput";
import { useSelector } from "react-redux/es/hooks/useSelector";
import ImageProcessingNotice from "../components/BoardDisplay/ImageProcessingNotice";
import BoardControlsContainer from "../components/BoardControls/BoardControlsContainer";
import BoardStats from "../components/BoardStats/BoardStats";
import WordTableContainer from "../components/WordTable/WordTableContainer";
import SelectedWordInfoPanel from "../components/SelectedWordInfo/SelectedWordInfoPanel";

function MainPage() {
  // Set up a selector for the boardDataSlice.
  const boardDataSlice = useSelector((state) => state.boardData);

  return (
    <div style={{ padding: "10px", textAlign: "center" }}>
      <Grid>
        <Grid.Col span={12}>
          {/* Header */}
          <div style={{ fontSize: "30px", fontWeight: "bold" }}>
            {/* 
          
            <img
              src="/images/logo.png"
              alt="A Boggle board with no filter applied."
              style={{
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
                width: 300
              }}
            />
          */}
            Boggle Vision
          </div>
        </Grid.Col>
        <Grid.Col span={12}>
          {/* Board Display */}
          <div>
            <div>
              <ImageInput />
            </div>
          </div>
        </Grid.Col>
        <Grid.Col span={12}>
          {boardDataSlice.boardData !== null ? (
            <div>
              <Grid.Col span={12}>
                {/* Board Controls */}
                <div>
                  <BoardControlsContainer />
                </div>
              </Grid.Col>
              <Grid.Col span={12}>
                {/* Board Stats */}
                <div>
                  <BoardStats />
                </div>
              </Grid.Col>
              <Grid.Col span={12}>
                {/* Word Table */}
                <div style={{ width: "100%" }}>
                  <WordTableContainer />
                </div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "row",
                    justifyContent: "center",
                    // alignItems: "center",
                    width: "100%",
                  }}
                >
                  <SelectedWordInfoPanel />
                </div>
              </Grid.Col>
            </div>
          ) : null}
        </Grid.Col>
      </Grid>
    </div>
  );
}
export default MainPage;
