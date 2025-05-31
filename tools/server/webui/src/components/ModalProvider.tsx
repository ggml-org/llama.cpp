import React, { createContext, useState, useContext } from 'react';

type ModalContextType = {
  showConfirm: (message: string) => Promise<boolean>;
  showPrompt: (
    message: string,
    defaultValue?: string
  ) => Promise<string | undefined>;
  showAlert: (message: string) => Promise<void>;
};
const ModalContext = createContext<ModalContextType>(null!);

interface ModalState<T> {
  isOpen: boolean;
  message: string;
  defaultValue?: string;
  resolve: ((value: T) => void) | null;
}

export function ModalProvider({ children }: { children: React.ReactNode }) {
  const [confirmState, setConfirmState] = useState<ModalState<boolean>>({
    isOpen: false,
    message: '',
    resolve: null,
  });
  const [promptState, setPromptState] = useState<
    ModalState<string | undefined>
  >({ isOpen: false, message: '', resolve: null });
  const [alertState, setAlertState] = useState<ModalState<void>>({
    isOpen: false,
    message: '',
    resolve: null,
  });
  const inputRef = React.useRef<HTMLInputElement>(null);

  const showConfirm = (message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmState({ isOpen: true, message, resolve });
    });
  };

  const showPrompt = (
    message: string,
    defaultValue?: string
  ): Promise<string | undefined> => {
    return new Promise((resolve) => {
      setPromptState({ isOpen: true, message, defaultValue, resolve });
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

  const handlePrompt = (result?: string) => {
    promptState.resolve?.(result);
    setPromptState({ isOpen: false, message: '', resolve: null });
  };

  const handleAlertClose = () => {
    alertState.resolve?.();
    setAlertState({ isOpen: false, message: '', resolve: null });
  };

  return (
    <ModalContext.Provider value={{ showConfirm, showPrompt, showAlert }}>
      {children}

      {/* Confirm Modal */}
      {confirmState.isOpen && (
        <dialog className="modal modal-open z-[1100]">
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
        </dialog>
      )}

      {/* Prompt Modal */}
      {promptState.isOpen && (
        <dialog className="modal modal-open z-[1100]">
          <div className="modal-box">
            <h3 className="font-bold text-lg">{promptState.message}</h3>
            <input
              type="text"
              className="input input-bordered w-full mt-2"
              defaultValue={promptState.defaultValue}
              ref={inputRef}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handlePrompt((e.target as HTMLInputElement).value);
                }
              }}
            />
            <div className="modal-action">
              <button className="btn btn-ghost" onClick={() => handlePrompt()}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={() => handlePrompt(inputRef.current?.value)}
              >
                Submit
              </button>
            </div>
          </div>
        </dialog>
      )}

      {/* Alert Modal */}
      {alertState.isOpen && (
        <dialog className="modal modal-open z-[1100]">
          <div className="modal-box">
            <h3 className="font-bold text-lg">{alertState.message}</h3>
            <div className="modal-action">
              <button className="btn" onClick={handleAlertClose}>
                OK
              </button>
            </div>
          </div>
        </dialog>
      )}
    </ModalContext.Provider>
  );
}

export function useModals() {
  const context = useContext(ModalContext);
  if (!context) throw new Error('useModals must be used within ModalProvider');
  return context;
}
