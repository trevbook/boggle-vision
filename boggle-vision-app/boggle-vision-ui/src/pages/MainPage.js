import { Provider } from "react-redux";
import { MantineProvider, Grid, Modal } from "@mantine/core";
import ImageInput from "../components/BoardDisplay/ImageInput";
import { useSelector } from "react-redux/es/hooks/useSelector";
import BoardControlsContainer from "../components/BoardControls/BoardControlsContainer";
import BoardStats from "../components/BoardStats/BoardStats";
import WordTableContainer from "../components/WordTable/WordTableContainer";
import SelectedWordInfoPanel from "../components/SelectedWordInfo/SelectedWordInfoPanel";
import { Loader } from "@mantine/core";

function MainPage() {
  // Set up a selector for the boardDataSlice.
  const boardDataSlice = useSelector((state) => state.boardData);

  const image_loading = useSelector((state) => state.imageUpload.loading);

  return (
    <div style={{ padding: "10px" }}>
      <Grid>
        <Grid.Col span={12}>
          {/* Header */}
          <div
            style={{
              fontSize: "30px",
              fontWeight: "bold",
              textAlign: "center",
            }}
          >
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
        {image_loading ? (
          <Grid.Col span={12}>
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                height: "60px",
                width: "100%",
                position: "absolute",
              }}
            >
              <Loader color={"blue"} />
            </div>
          </Grid.Col>
        ) : (
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
                      width: "100%",
                    }}
                  >
                    <SelectedWordInfoPanel />
                  </div>
                </Grid.Col>
              </div>
            ) : null}
          </Grid.Col>
        )}
      </Grid>
    </div>
  );
}
export default MainPage;
