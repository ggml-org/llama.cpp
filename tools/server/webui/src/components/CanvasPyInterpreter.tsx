import { useEffect, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import { OpenInNewTab, XCloseButton } from '../utils/common';
import { CanvasType } from '../utils/types';
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline';
import StorageUtils from '../utils/storage';

const canInterrupt = typeof SharedArrayBuffer === 'function';

// adapted from https://pyodide.org/en/stable/usage/webworker.html
const WORKER_CODE = `
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");

let stdOutAndErr = [];

let pyodideReadyPromise = loadPyodide({
  stdout: (data) => stdOutAndErr.push(data),
  stderr: (data) => stdOutAndErr.push(data),
});

let alreadySetBuff = false;

self.onmessage = async (event) => {
  stdOutAndErr = [];

  // make sure loading is done
  const pyodide = await pyodideReadyPromise;
  const { id, python, context, interruptBuffer } = event.data;

  if (interruptBuffer && !alreadySetBuff) {
    pyodide.setInterruptBuffer(interruptBuffer);
    alreadySetBuff = true;
  }

  // Now load any packages we need, run the code, and send the result back.
  await pyodide.loadPackagesFromImports(python);

  // make a Python dictionary with the data from content
  const dict = pyodide.globals.get("dict");
  const globals = dict(Object.entries(context));
  try {
    self.postMessage({ id, running: true });
    // Execute the python code in this context
    const result = pyodide.runPython(python, { globals });
    self.postMessage({ result, id, stdOutAndErr });
  } catch (error) {
    self.postMessage({ error: error.message, id });
  }
  interruptBuffer[0] = 0;
};
`;

let worker: Worker;
const interruptBuffer = canInterrupt
  ? new Uint8Array(new SharedArrayBuffer(1))
  : null;

const startWorker = () => {
  if (!worker) {
    worker = new Worker(
      URL.createObjectURL(new Blob([WORKER_CODE], { type: 'text/javascript' }))
    );
  }
};

if (StorageUtils.getConfig().pyIntepreterEnabled) {
  startWorker();
}

const runCodeInWorker = (
  pyCode: string,
  callbackRunning: () => void
): {
  donePromise: Promise<string>;
  interrupt: () => void;
} => {
  startWorker();
  const id = Math.random() * 1e8;
  const context = {};
  if (interruptBuffer) {
    interruptBuffer[0] = 0;
  }

  const donePromise = new Promise<string>((resolve) => {
    worker.onmessage = (event) => {
      const { error, stdOutAndErr, running } = event.data;
      if (id !== event.data.id) return;
      if (running) {
        callbackRunning();
        return;
      } else if (error) {
        resolve(error.toString());
      } else {
        resolve(stdOutAndErr.join('\n'));
      }
    };
    worker.postMessage({ id, python: pyCode, context, interruptBuffer });
  });

  const interrupt = () => {
    console.log('Interrupting...');
    console.trace();
    if (interruptBuffer) {
      interruptBuffer[0] = 2;
    }
  };

  return { donePromise, interrupt };
};

export default function CanvasPyInterpreter() {
  const { canvasData, setCanvasData } = useAppContext();

  const [code, setCode] = useState(canvasData?.content ?? ''); // copy to avoid direct mutation
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState('');
  const [interruptFn, setInterruptFn] = useState<() => void>();
  const [showStopBtn, setShowStopBtn] = useState(false);

  const runCode = async (pycode: string) => {
    interruptFn?.();
    setRunning(true);
    setOutput('Loading Pyodide...');
    const { donePromise, interrupt } = runCodeInWorker(pycode, () => {
      setOutput('Running...');
      setShowStopBtn(canInterrupt);
    });
    setInterruptFn(() => interrupt);
    const out = await donePromise;
    setOutput(out);
    setRunning(false);
    setShowStopBtn(false);
  };

  // run code on mount
  useEffect(() => {
    setCode(canvasData?.content ?? '');
    runCode(canvasData?.content ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasData?.content]);

  if (canvasData?.type !== CanvasType.PY_INTERPRETER) {
    return null;
  }

  return (
    <div className="card bg-base-200 w-full h-full shadow-xl">
      <div className="card-body">
        <div className="flex justify-between items-center mb-4">
          <span className="text-lg font-bold">Python Interpreter</span>
          <XCloseButton
            className="bg-base-100"
            onClick={() => setCanvasData(null)}
          />
        </div>
        <div className="grid grid-rows-3 gap-4 h-full">
          <textarea
            className="textarea textarea-bordered w-full h-full font-mono"
            value={code}
            onChange={(e) => setCode(e.target.value)}
          ></textarea>
          <div className="font-mono flex flex-col row-span-2">
            <div className="flex items-center mb-2">
              <button
                className="btn btn-sm bg-base-100"
                onClick={() => runCode(code)}
                disabled={running}
              >
                <PlayIcon className="h-6 w-6" /> Run
              </button>
              {showStopBtn && (
                <button
                  className="btn btn-sm bg-base-100 ml-2"
                  onClick={() => interruptFn?.()}
                >
                  <StopIcon className="h-6 w-6" /> Stop
                </button>
              )}
              <span className="grow text-right text-xs">
                <OpenInNewTab href="https://github.com/ggerganov/llama.cpp/issues/11762">
                  Report a bug
                </OpenInNewTab>
              </span>
            </div>
            <textarea
              className="textarea textarea-bordered h-full dark-color"
              value={output}
              readOnly
            ></textarea>
          </div>
        </div>
      </div>
    </div>
  );
}
