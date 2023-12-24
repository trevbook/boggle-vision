// import './App.css';
import { Provider } from "react-redux";
import { MantineProvider } from "@mantine/core";
import store from "./store";
import MainPage from "./pages/MainPage";

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
