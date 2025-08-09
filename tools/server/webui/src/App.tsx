import { HashRouter, Outlet, Route, Routes } from 'react-router';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import { AppContextProvider, useAppContext } from './utils/app.context';
import ChatScreen from './components/ChatScreen';
import SettingDialog from './components/SettingDialog';
import { Toaster } from 'react-hot-toast';
import { ModalProvider } from './components/ModalProvider';

function App() {
  return (
    <ModalProvider>
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
    </ModalProvider>
  );
}

function AppLayout() {
  const { showSettings, setShowSettings } = useAppContext();
  return (
    <>
      <Sidebar />
      <main
        className="drawer-content grow flex flex-col h-screen mx-auto px-4 overflow-auto bg-base-100"
        id="main-scroll"
      >
        <Header />
        <Outlet />
        <ServerInfo />
      </main>
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

function ServerInfo() {
  const { serverProps } = useAppContext();
  const modalities = [];
  if (serverProps?.modalities?.audio) {
    modalities.push('audio');
  }
  if (serverProps?.modalities?.vision) {
    modalities.push('vision');
  }
  return (
    <div
      className="sticky bottom-0 w-full pb-1 text-base-content/70 text-xs text-center"
      tabIndex={0}
      aria-description="Server information"
    >
      <span>
        <b>Llama.cpp</b> {serverProps?.build_info}
      </span>

      <span className="sm:ml-2">
        {modalities.length > 0 ? (
          <>
            <br className="sm:hidden" />
            <b>Supported modalities:</b> {modalities.join(', ')}
          </>
        ) : (
          ''
        )}
      </span>
    </div>
  );
}

export default App;
