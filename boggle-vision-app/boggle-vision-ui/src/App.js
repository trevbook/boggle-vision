import './App.css';
import { Provider } from 'react-redux'
import { MantineProvider, Grid } from "@mantine/core";
import ImageInput from './components/ImageInput';
import store from './store';
import { useSelector } from 'react-redux/es/hooks/useSelector';
import ImageProcessingNotice from './components/ImageProcessingNotice';
import { setImage } from './slices/imageUploadSlice';


function App() {

  return (
    <Provider store={store}>
      <MantineProvider>
        <div style={{ "padding": "10px", "textAlign": "center" }}>
          <Grid>
            <Grid.Col span={12}>
              <h1>Boggle Vision</h1>
              <div>
                <ImageInput />
              </div>
              <div>
                <ImageProcessingNotice />
              </div>
            </Grid.Col>
          </Grid>
        </div>
      </MantineProvider>
    </Provider>

  );
}

export default App;
