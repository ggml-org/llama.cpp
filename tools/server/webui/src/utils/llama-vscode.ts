import { useEffect } from 'react';
import { ChatTextareaApi } from '../components/useChatTextarea.ts';
import { ChatExtraContextApi } from '../components/useChatExtraContext.tsx';

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

export const useVSCodeContext = (
  textarea: ChatTextareaApi,
  extraContext: ChatExtraContextApi
) => {
  // Accept setText message from a parent window and set inputMsg and extraContext
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.command === 'setText') {
        const data: SetTextEvData = event.data;
        textarea.setValue(data?.text);
        if (data?.context && data.context.length > 0) {
          extraContext.clearItems();
          extraContext.addItems([
            {
              type: 'context',
              name: 'Extra context',
              content: data.context,
            },
          ]);
        }
        textarea.focus();
        setTimeout(() => {
          textarea.refOnSubmit.current?.();
        }, 10); // wait for setExtraContext to finish
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [textarea, extraContext]);

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

  return {};
};
