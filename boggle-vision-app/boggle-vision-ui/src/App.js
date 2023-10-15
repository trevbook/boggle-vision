// import './App.css';
import { Provider } from 'react-redux'
import { MantineProvider, Grid, Modal } from "@mantine/core";
import ImageInput from './components/BoardDisplay/ImageInput';
import store from './store';
import { useSelector } from 'react-redux/es/hooks/useSelector';
import ImageProcessingNotice from './components/BoardDisplay/ImageProcessingNotice';
import { setImage } from './slices/imageUploadSlice';
import BoardControlsContainer from './components/BoardControls/BoardControlsContainer';
import BoardStats from './components/BoardStats/BoardStats';
import WordTableContainer from './components/WordTable/WordTableContainer';
import { UseSelector } from 'react-redux/es/hooks/useSelector';
import MainPage from './pages/MainPage';
import '@mantine/core/styles.css';

function App() {

  return (
    <Provider store={store}>
      <MantineProvider>
        <MainPage />
      </MantineProvider>
    </Provider>

  );
}

export default App;
