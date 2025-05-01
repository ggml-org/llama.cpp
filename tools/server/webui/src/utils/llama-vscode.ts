import { useEffect, useState } from 'react';
import { MessageExtraContext } from './types';
import { ChatTextareaApi } from '../components/useChatTextarea.ts';

// Extra context when using llama.cpp WebUI from llama-vscode, inside an iframe
// Ref: https://github.com/ggml-org/llama.cpp/pull/11940

interface SetTextEvData {
  text: string;
  context: string;
}

/**
 * To test it:
 * window.postMessage({ command: 'setText', text: 'Spot the syntax error', context: 'def test()\n  return 123' }, '*');
 */

export const useVSCodeContext = (textarea: ChatTextareaApi) => {
  const [extraContext, setExtraContext] = useState<MessageExtraContext | null>(
    null
  );

  // Accept setText message from a parent window and set inputMsg and extraContext
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.command === 'setText') {
        const data: SetTextEvData = event.data;
        textarea.setValue(data?.text);
        if (data?.context && data.context.length > 0) {
          setExtraContext({
            type: 'context',
            content: data.context,
          });
        }
        textarea.focus();
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [textarea]);

  // Add a keydown listener that sends the "escapePressed" message to the parent window
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        window.parent.postMessage({ command: 'escapePressed' }, '*');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return {
    extraContext,
    // call once the user message is sent, to clear the extra context
    clearExtraContext: () => setExtraContext(null),
  };
};
