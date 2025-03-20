import { useEffect, useMemo, useRef, useState } from 'react';
import { CallbackGeneratedChunk, useAppContext } from '../utils/app.context';
import ChatMessage from './ChatMessage';
import { CanvasType, Message, PendingMessage } from '../utils/types';
import { classNames, cleanCurrentUrl, throttle } from '../utils/misc';
import CanvasPyInterpreter from './CanvasPyInterpreter';
import StorageUtils from '../utils/storage';
import { useVSCodeContext } from '../utils/llama-vscode';

/**
 * A message display is a message node with additional information for rendering.
 * For example, siblings of the message node are stored as their last node (aka leaf node).
 */
export interface MessageDisplay {
  msg: Message | PendingMessage;
  siblingLeafNodeIds: Message['id'][];
  siblingCurrIdx: number;
  isPending?: boolean;
}

/**
 * If the current URL contains "?m=...", prefill the message input with the value.
 * If the current URL contains "?q=...", prefill and SEND the message.
 */
const prefilledMsg = {
  content() {
    const url = new URL(window.location.href);
    return url.searchParams.get('m') ?? url.searchParams.get('q') ?? '';
  },
  shouldSend() {
    const url = new URL(window.location.href);
    return url.searchParams.has('q');
  },
  clear() {
    cleanCurrentUrl(['m', 'q']);
  },
};

function getListMessageDisplay(
  msgs: Readonly<Message[]>,
  leafNodeId: Message['id']
): MessageDisplay[] {
  const currNodes = StorageUtils.filterByLeafNodeId(msgs, leafNodeId, true);
  const res: MessageDisplay[] = [];
  const nodeMap = new Map<Message['id'], Message>();
  for (const msg of msgs) {
    nodeMap.set(msg.id, msg);
  }
  // find leaf node from a message node
  const findLeafNode = (msgId: Message['id']): Message['id'] => {
    let currNode: Message | undefined = nodeMap.get(msgId);
    while (currNode) {
      if (currNode.children.length === 0) break;
      currNode = nodeMap.get(currNode.children.at(-1) ?? -1);
    }
    return currNode?.id ?? -1;
  };
  // traverse the current nodes
  for (const msg of currNodes) {
    const parentNode = nodeMap.get(msg.parent ?? -1);
    if (!parentNode) continue;
    const siblings = parentNode.children;
    if (msg.type !== 'root') {
      res.push({
        msg,
        siblingLeafNodeIds: siblings.map(findLeafNode),
        siblingCurrIdx: siblings.indexOf(msg.id),
      });
    }
  }
  return res;
}

const scrollToBottom = throttle(
  (requiresNearBottom: boolean, delay: number = 80) => {
    const mainScrollElem = document.getElementById('main-scroll');
    if (!mainScrollElem) return;
    const spaceToBottom =
      mainScrollElem.scrollHeight -
      mainScrollElem.scrollTop -
      mainScrollElem.clientHeight;
    if (!requiresNearBottom || spaceToBottom < 50) {
      setTimeout(
        () => mainScrollElem.scrollTo({ top: mainScrollElem.scrollHeight }),
        delay
      );
    }
  },
  80
);

export default function ChatScreen() {
  const {
    viewingChat,
    sendMessage,
    isGenerating,
    stopGenerating,
    pendingMessages,
    canvasData,
    replaceMessageAndGenerate,
  } = useAppContext();
  const textarea = useOptimizedTextarea(prefilledMsg.content());

  const { extraContext, clearExtraContext } = useVSCodeContext(textarea);
  // TODO: improve this when we have "upload file" feature
  const currExtra: Message['extra'] = extraContext ? [extraContext] : undefined;

  // keep track of leaf node for rendering
  const [currNodeId, setCurrNodeId] = useState<number>(-1);
  const messages: MessageDisplay[] = useMemo(() => {
    if (!viewingChat) return [];
    else return getListMessageDisplay(viewingChat.messages, currNodeId);
  }, [currNodeId, viewingChat]);

  const currConvId = viewingChat?.conv.id ?? null;
  const pendingMsg: PendingMessage | undefined =
    pendingMessages[currConvId ?? ''];

  useEffect(() => {
    // reset to latest node when conversation changes
    setCurrNodeId(-1);
    // scroll to bottom when conversation changes
    scrollToBottom(false, 1);
  }, [currConvId]);

  const onChunk: CallbackGeneratedChunk = (currLeafNodeId?: Message['id']) => {
    if (currLeafNodeId) {
      setCurrNodeId(currLeafNodeId);
    }
    scrollToBottom(true);
  };

  const sendNewMessage = async () => {
    const lastInpMsg = textarea.value();
    if (lastInpMsg.trim().length === 0 || isGenerating(currConvId ?? ''))
      return;
    textarea.setValue('');
    scrollToBottom(false);
    setCurrNodeId(-1);
    // get the last message node
    const lastMsgNodeId = messages.at(-1)?.msg.id ?? null;
    if (
      !(await sendMessage(
        currConvId,
        lastMsgNodeId,
        lastInpMsg,
        currExtra,
        onChunk
      ))
    ) {
      // restore the input message if failed
      textarea.setValue(lastInpMsg);
    }
    // OK
    clearExtraContext();
  };

  const handleEditMessage = async (msg: Message, content: string) => {
    if (!viewingChat) return;
    setCurrNodeId(msg.id);
    scrollToBottom(false);
    await replaceMessageAndGenerate(
      viewingChat.conv.id,
      msg.parent,
      content,
      msg.extra,
      onChunk
    );
    setCurrNodeId(-1);
    scrollToBottom(false);
  };

  const handleRegenerateMessage = async (msg: Message) => {
    if (!viewingChat) return;
    setCurrNodeId(msg.parent);
    scrollToBottom(false);
    await replaceMessageAndGenerate(
      viewingChat.conv.id,
      msg.parent,
      null,
      msg.extra,
      onChunk
    );
    setCurrNodeId(-1);
    scrollToBottom(false);
  };

  const hasCanvas = !!canvasData;

  useEffect(() => {
    if (prefilledMsg.shouldSend()) {
      // send the prefilled message if needed
      sendNewMessage();
    } else {
      // otherwise, focus on the input
      textarea.focus();
    }
    prefilledMsg.clear();
    // no need to keep track of sendNewMessage
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [textarea.ref]);

  // due to some timing issues of StorageUtils.appendMsg(), we need to make sure the pendingMsg is not duplicated upon rendering (i.e. appears once in the saved conversation and once in the pendingMsg)
  const pendingMsgDisplay: MessageDisplay[] =
    pendingMsg && messages.at(-1)?.msg.id !== pendingMsg.id
      ? [
          {
            msg: pendingMsg,
            siblingLeafNodeIds: [],
            siblingCurrIdx: 0,
            isPending: true,
          },
        ]
      : [];

  return (
    <div
      className={classNames({
        'grid lg:gap-8 grow transition-[300ms]': true,
        'grid-cols-[1fr_0fr] lg:grid-cols-[1fr_1fr]': hasCanvas, // adapted for mobile
        'grid-cols-[1fr_0fr]': !hasCanvas,
      })}
    >
      <div
        className={classNames({
          'flex flex-col w-full max-w-[900px] mx-auto': true,
          'hidden lg:flex': hasCanvas, // adapted for mobile
          flex: !hasCanvas,
        })}
      >
        {/* chat messages */}
        <div id="messages-list" className="grow">
          <div className="mt-auto flex justify-center">
            {/* placeholder to shift the message to the bottom */}
            {viewingChat ? '' : 'Send a message to start'}
          </div>
          {[...messages, ...pendingMsgDisplay].map((msg) => (
            <ChatMessage
              key={msg.msg.id}
              msg={msg.msg}
              siblingLeafNodeIds={msg.siblingLeafNodeIds}
              siblingCurrIdx={msg.siblingCurrIdx}
              onRegenerateMessage={handleRegenerateMessage}
              onEditMessage={handleEditMessage}
              onChangeSibling={setCurrNodeId}
            />
          ))}
        </div>

        {/* chat input */}
        <div className="flex flex-row items-center pt-8 pb-6 sticky bottom-0 bg-base-100">
          <textarea
            className="textarea textarea-bordered w-full"
            placeholder="Type a message (Shift+Enter to add a new line)"
            ref={textarea.ref}
            onKeyDown={(e) => {
              if (e.nativeEvent.isComposing || e.keyCode === 229) return;
              if (e.key === 'Enter' && e.shiftKey) return;
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendNewMessage();
              }
            }}
            id="msg-input"
            dir="auto"
          ></textarea>
          {isGenerating(currConvId ?? '') ? (
            <button
              className="btn btn-neutral ml-2"
              onClick={() => stopGenerating(currConvId ?? '')}
            >
              Stop
            </button>
          ) : (
            <button className="btn btn-primary ml-2" onClick={sendNewMessage}>
              Send
            </button>
          )}
        </div>
      </div>
      <div className="w-full sticky top-[7em] h-[calc(100vh-9em)]">
        {canvasData?.type === CanvasType.PY_INTERPRETER && (
          <CanvasPyInterpreter />
        )}
      </div>
    </div>
  );
}

export interface OptimizedTextareaValue {
  value: () => string;
  setValue: (value: string) => void;
  focus: () => void;
  ref: React.RefObject<HTMLTextAreaElement>;
}

// This is a workaround to prevent the textarea from re-rendering when the inner content changes
// See https://github.com/ggml-org/llama.cpp/pull/12299
function useOptimizedTextarea(initValue: string): OptimizedTextareaValue {
  const [savedInitValue, setSavedInitValue] = useState<string>(initValue);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current && savedInitValue) {
      textareaRef.current.value = savedInitValue;
      setSavedInitValue('');
    }
  }, [textareaRef, savedInitValue, setSavedInitValue]);

  return {
    value: () => {
      return textareaRef.current?.value ?? savedInitValue;
    },
    setValue: (value: string) => {
      if (textareaRef.current) {
        textareaRef.current.value = value;
      }
    },
    focus: () => {
      if (textareaRef.current) {
        // focus and move the cursor to the end
        textareaRef.current.focus();
        textareaRef.current.selectionStart = textareaRef.current.value.length;
      }
    },
    ref: textareaRef,
  };
}
