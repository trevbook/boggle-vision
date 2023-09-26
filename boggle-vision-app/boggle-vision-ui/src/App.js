import logo from './logo.svg';
import './App.css';
import { MantineProvider } from "@mantine/core";
import ImageInputComponent from './components/ImageInputComponent';

function App() {
  return (
    <MantineProvider>
      <div style={{ "padding": "10px" }}>
        <h1>Boggle Vision</h1>
        <div>
          <p>Below, take a picture of your Boggle board.</p>
          <ImageInputComponent />
        </div>
      </div>
    </MantineProvider>
  );
}

export default App;
