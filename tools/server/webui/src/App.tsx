import { HashRouter, Outlet, Route, Routes } from 'react-router';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import { AppContextProvider, useAppContext } from './utils/app.context';
import ChatScreen from './components/ChatScreen';
import SettingDialog from './components/SettingDialog';
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <HashRouter>
      <div className="flex flex-row drawer lg:drawer-open">
        <AppContextProvider>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/chat/:convId" element={<ChatScreen />} />
              <Route path="*" element={<ChatScreen />} />
            </Route>
          </Routes>
        </AppContextProvider>
      </div>
    </HashRouter>
  );
}

function AppLayout() {
  const { showSettings, setShowSettings } = useAppContext();
  return (
    <>
      <Sidebar />
      <div
        className="drawer-content grow flex flex-col h-screen w-screen mx-auto px-4 overflow-auto bg-base-100"
        id="main-scroll"
      >
        <Header />
        <Outlet />
      </div>
      {
        <SettingDialog
          show={showSettings}
          onClose={() => setShowSettings(false)}
        />
      }
      <Toaster />
    </>
  );
}

export default App;
