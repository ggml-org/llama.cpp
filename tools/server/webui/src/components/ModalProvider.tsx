import React, { createContext, useState, useContext } from 'react';

type ModalContextType = {
  showConfirm: (message: string) => Promise<boolean>;
  showAlert: (message: string) => Promise<void>;
};
const ModalContext = createContext<ModalContextType>(null!);

export function ModalProvider({ children }: { children: React.ReactNode }) {
  const [confirmState, setConfirmState] = useState<{
    isOpen: boolean;
    message: string;
    resolve: ((value: boolean) => void) | null;
  }>({ isOpen: false, message: '', resolve: null });

  const [alertState, setAlertState] = useState<{
    isOpen: boolean;
    message: string;
    resolve: (() => void) | null;
  }>({ isOpen: false, message: '', resolve: null });

  const showConfirm = (message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmState({ isOpen: true, message, resolve });
    });
  };

  const showAlert = (message: string): Promise<void> => {
    return new Promise((resolve) => {
      setAlertState({ isOpen: true, message, resolve });
    });
  };

  const handleConfirm = (result: boolean) => {
    confirmState.resolve?.(result);
    setConfirmState({ isOpen: false, message: '', resolve: null });
  };

  const handleAlertClose = () => {
    alertState.resolve?.();
    setAlertState({ isOpen: false, message: '', resolve: null });
  };

  return (
    <ModalContext.Provider value={{ showConfirm, showAlert }}>
      {children}

      {/* Confirm Modal */}
      {confirmState.isOpen && (
        <div className="modal modal-open z-[1100]">
          <div className="modal-box">
            <h3 className="font-bold text-lg">{confirmState.message}</h3>
            <div className="modal-action">
              <button
                className="btn btn-ghost"
                onClick={() => handleConfirm(false)}
              >
                Cancel
              </button>
              <button
                className="btn btn-error"
                onClick={() => handleConfirm(true)}
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Alert Modal */}
      {alertState.isOpen && (
        <div className="modal modal-open z-[1100]">
          <div className="modal-box">
            <h3 className="font-bold text-lg">{alertState.message}</h3>
            <div className="modal-action">
              <button className="btn" onClick={handleAlertClose}>
                OK
              </button>
            </div>
          </div>
        </div>
      )}
    </ModalContext.Provider>
  );
}

export function useModals() {
  const context = useContext(ModalContext);
  if (!context) throw new Error('useModals must be used within ModalProvider');
  return context;
}
