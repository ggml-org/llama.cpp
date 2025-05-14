import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  APIMessage,
  CanvasData,
  Conversation,
  LlamaCppServerProps,
  Message,
  PendingMessage,
  ViewingChat,
} from './types';
import StorageUtils from './storage';
import {
  filterThoughtFromMsgs,
  normalizeMsgsForAPI,
  getSSEStreamAsync,
  getServerProps,
} from './misc';
import { BASE_URL, CONFIG_DEFAULT, isDev } from '../Config';
import { matchPath, useLocation, useNavigate } from 'react-router';
import toast from 'react-hot-toast';

interface AppContextValue {
  // conversations and messages
  viewingChat: ViewingChat | null;
  pendingMessages: Record<Conversation['id'], PendingMessage>;
  isGenerating: (convId: string) => boolean;
  sendMessage: (
    convId: string | null,
    leafNodeId: Message['id'] | null,
    content: string,
    extra: Message['extra'],
    onChunk: CallbackGeneratedChunk
  ) => Promise<boolean>;
  stopGenerating: (convId: string) => void;
  replaceMessageAndGenerate: (
    convId: string,
    parentNodeId: Message['id'], // the parent node of the message to be replaced
    content: string | null,
    extra: Message['extra'],
    onChunk: CallbackGeneratedChunk
  ) => Promise<void>;
  continueMessageAndGenerate: (
    convId: string,
    messageIdToContinue: Message['id'],
    newContent: string,
    onChunk: CallbackGeneratedChunk
  ) => Promise<void>;

  // canvas
  canvasData: CanvasData | null;
  setCanvasData: (data: CanvasData | null) => void;

  // config
  config: typeof CONFIG_DEFAULT;
  saveConfig: (config: typeof CONFIG_DEFAULT) => void;
  showSettings: boolean;
  setShowSettings: (show: boolean) => void;

  // props
  serverProps: LlamaCppServerProps | null;
}

// this callback is used for scrolling to the bottom of the chat and switching to the last node
export type CallbackGeneratedChunk = (currLeafNodeId?: Message['id']) => void;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const AppContext = createContext<AppContextValue>({} as any);

const getViewingChat = async (convId: string): Promise<ViewingChat | null> => {
  const conv = await StorageUtils.getOneConversation(convId);
  if (!conv) return null;
  return {
    conv: conv,
    // all messages from all branches, not filtered by last node
    messages: await StorageUtils.getMessages(convId),
  };
};

export const AppContextProvider = ({
  children,
}: {
  children: React.ReactElement;
}) => {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const params = matchPath('/chat/:convId', pathname);
  const convId = params?.params?.convId;

  const [serverProps, setServerProps] = useState<LlamaCppServerProps | null>(
    null
  );
  const [viewingChat, setViewingChat] = useState<ViewingChat | null>(null);
  const [pendingMessages, setPendingMessages] = useState<
    Record<Conversation['id'], PendingMessage>
  >({});
  const [aborts, setAborts] = useState<
    Record<Conversation['id'], AbortController>
  >({});
  const [config, setConfig] = useState(StorageUtils.getConfig());
  const [canvasData, setCanvasData] = useState<CanvasData | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // get server props
  useEffect(() => {
    getServerProps(BASE_URL, config.apiKey)
      .then((props) => {
        console.debug('Server props:', props);
        setServerProps(props);
      })
      .catch((err) => {
        console.error(err);
        toast.error('Failed to fetch server props');
      });
    // eslint-disable-next-line
  }, []);

  // handle change when the convId from URL is changed
  useEffect(() => {
    // also reset the canvas data
    setCanvasData(null);
    const handleConversationChange = async (changedConvId: string) => {
      if (changedConvId !== convId) return;
      setViewingChat(await getViewingChat(changedConvId));
    };
    StorageUtils.onConversationChanged(handleConversationChange);
    getViewingChat(convId ?? '').then(setViewingChat);
    return () => {
      StorageUtils.offConversationChanged(handleConversationChange);
    };
  }, [convId]);

  const setPending = (convId: string, pendingMsg: PendingMessage | null) => {
    // if pendingMsg is null, remove the key from the object
    if (!pendingMsg) {
      setPendingMessages((prev) => {
        const newState = { ...prev };
        delete newState[convId];
        return newState;
      });
    } else {
      setPendingMessages((prev) => ({ ...prev, [convId]: pendingMsg }));
    }
  };

  const setAbort = (convId: string, controller: AbortController | null) => {
    if (!controller) {
      setAborts((prev) => {
        const newState = { ...prev };
        delete newState[convId];
        return newState;
      });
    } else {
      setAborts((prev) => ({ ...prev, [convId]: controller }));
    }
  };

  ////////////////////////////////////////////////////////////////////////
  // public functions

  const isGenerating = (convId: string) => !!pendingMessages[convId];

  const generateMessage = async (
    convId: string,
    leafNodeId: Message['id'],
    onChunk: CallbackGeneratedChunk,
    isContinuation: boolean = false
  ) => {
    if (isGenerating(convId)) return;

    const config = StorageUtils.getConfig();
    const currConversation = await StorageUtils.getOneConversation(convId);
    if (!currConversation) {
      throw new Error('Current conversation is not found');
    }

    const currMessages = StorageUtils.filterByLeafNodeId(
      await StorageUtils.getMessages(convId),
      leafNodeId,
      false
    );
    const abortController = new AbortController();
    setAbort(convId, abortController);

    if (!currMessages) {
      throw new Error('Current messages are not found');
    }

    const pendingId = Date.now() + 1;
    let pendingMsg: Message | PendingMessage;

    if (isContinuation) {
      const existingAsstMsg = await StorageUtils.getMessage(convId, leafNodeId);
      if (!existingAsstMsg || existingAsstMsg.role !== 'assistant') {
        toast.error(
          'Cannot continue: target message not found or not an assistant message.'
        );
        throw new Error(
          'Cannot continue: target message not found or not an assistant message.'
        );
      }
      pendingMsg = {
        ...existingAsstMsg,
        content: existingAsstMsg.content || '',
      };
      setPending(convId, pendingMsg as PendingMessage);
    } else {
      pendingMsg = {
        id: pendingId,
        convId,
        type: 'text',
        timestamp: pendingId,
        role: 'assistant',
        content: null,
        parent: leafNodeId,
        children: [],
      };
      setPending(convId, pendingMsg as PendingMessage);
    }

    try {
      // prepare messages for API
      let messages: APIMessage[] = [
        ...(config.systemMessage.length === 0
          ? []
          : [{ role: 'system', content: config.systemMessage } as APIMessage]),
        ...normalizeMsgsForAPI(currMessages),
      ];
      if (config.excludeThoughtOnReq) {
        messages = filterThoughtFromMsgs(messages);
      }
      if (isDev) console.log({ messages });

      // prepare params
      const params = {
        messages,
        stream: true,
        cache_prompt: true,
        samplers: config.samplers,
        temperature: config.temperature,
        dynatemp_range: config.dynatemp_range,
        dynatemp_exponent: config.dynatemp_exponent,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: config.min_p,
        typical_p: config.typical_p,
        xtc_probability: config.xtc_probability,
        xtc_threshold: config.xtc_threshold,
        repeat_last_n: config.repeat_last_n,
        repeat_penalty: config.repeat_penalty,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        dry_multiplier: config.dry_multiplier,
        dry_base: config.dry_base,
        dry_allowed_length: config.dry_allowed_length,
        dry_penalty_last_n: config.dry_penalty_last_n,
        max_tokens: config.max_tokens,
        timings_per_token: !!config.showTokensPerSecond,
        ...(config.custom.length ? JSON.parse(config.custom) : {}),
      };

      // send request
      const fetchResponse = await fetch(`${BASE_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.apiKey
            ? { Authorization: `Bearer ${config.apiKey}` }
            : {}),
        },
        body: JSON.stringify(params),
        signal: abortController.signal,
      });
      if (fetchResponse.status !== 200) {
        const body = await fetchResponse.json();
        throw new Error(body?.error?.message || 'Unknown error');
      }
      const chunks = getSSEStreamAsync(fetchResponse);
      for await (const chunk of chunks) {
        // const stop = chunk.stop;
        if (chunk.error) {
          throw new Error(chunk.error?.message || 'Unknown error');
        }
        const addedContent = chunk.choices[0].delta.content;
        const lastContent = pendingMsg.content || '';
        if (addedContent) {
          pendingMsg = {
            ...pendingMsg,
            content: lastContent + addedContent,
          };
        }
        const timings = chunk.timings;
        if (timings && config.showTokensPerSecond) {
          // only extract what's really needed, to save some space
          pendingMsg.timings = {
            prompt_n: timings.prompt_n,
            prompt_ms: timings.prompt_ms,
            predicted_n: timings.predicted_n,
            predicted_ms: timings.predicted_ms,
          };
        }
        setPending(convId, pendingMsg as PendingMessage);
        onChunk(); // don't need to switch node for pending message
      }
    } catch (err) {
      setPending(convId, null);
      if ((err as Error).name === 'AbortError') {
        // user stopped the generation via stopGeneration() function
        // we can safely ignore this error
      } else {
        console.error(err);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        toast.error((err as any)?.message ?? 'Unknown error');
        throw err; // rethrow
      }
    }

    if (pendingMsg.content !== null) {
      if (isContinuation) {
        await StorageUtils.updateMessage(pendingMsg as Message);
      } else if (pendingMsg.content.trim().length > 0) {
        await StorageUtils.appendMsg(pendingMsg as Message, leafNodeId);
      }
    }
    setPending(convId, null);
    const finalNodeId = (pendingMsg as Message).id;
    onChunk(finalNodeId);
  };

  const sendMessage = async (
    convId: string | null,
    leafNodeId: Message['id'] | null,
    content: string,
    extra: Message['extra'],
    onChunk: CallbackGeneratedChunk
  ): Promise<boolean> => {
    if (isGenerating(convId ?? '') || content.trim().length === 0) return false;

    if (convId === null || convId.length === 0 || leafNodeId === null) {
      const conv = await StorageUtils.createConversation(
        content.substring(0, 256)
      );
      convId = conv.id;
      leafNodeId = conv.currNode;
      // if user is creating a new conversation, redirect to the new conversation
      navigate(`/chat/${convId}`);
    }

    const now = Date.now();
    const currMsgId = now;
    StorageUtils.appendMsg(
      {
        id: currMsgId,
        timestamp: now,
        type: 'text',
        convId,
        role: 'user',
        content,
        extra,
        parent: leafNodeId,
        children: [],
      },
      leafNodeId
    );
    onChunk(currMsgId);

    try {
      await generateMessage(convId, currMsgId, onChunk, false);
      return true;
    } catch (_) {
      // TODO: rollback
    }
    return false;
  };

  const stopGenerating = (convId: string) => {
    setPending(convId, null);
    aborts[convId]?.abort();
  };

  // if content is undefined, we remove last assistant message
  const replaceMessageAndGenerate = async (
    convId: string,
    parentNodeId: Message['id'], // the parent node of the message to be replaced
    content: string | null,
    extra: Message['extra'],
    onChunk: CallbackGeneratedChunk
  ) => {
    if (isGenerating(convId)) return;

    if (content !== null) {
      const now = Date.now();
      const currMsgId = now;
      StorageUtils.appendMsg(
        {
          id: currMsgId,
          timestamp: now,
          type: 'text',
          convId,
          role: 'user',
          content,
          extra,
          parent: parentNodeId,
          children: [],
        },
        parentNodeId
      );
      parentNodeId = currMsgId;
    }
    onChunk(parentNodeId);

    await generateMessage(convId, parentNodeId, onChunk);
  };

  const continueMessageAndGenerate = async (
    convId: string,
    messageIdToContinue: Message['id'],
    newContent: string,
    onChunk: CallbackGeneratedChunk
  ) => {
    if (isGenerating(convId)) return;

    const existingMessage = await StorageUtils.getMessage(
      convId,
      messageIdToContinue
    );
    if (!existingMessage || existingMessage.role !== 'assistant') {
      console.error(
        'Cannot continue non-assistant message or message not found'
      );
      toast.error(
        'Failed to continue message: Not an assistant message or not found.'
      );
      return;
    }

    const updatedAssistantMessage: Message = {
      ...existingMessage,
      content: newContent,
      timestamp: Date.now(),
      children: [], // Clear existing children to start a new branch of generation
      extra: existingMessage.extra, // Preserve existing extra data
    };

    await StorageUtils.updateMessage(updatedAssistantMessage);

    onChunk(messageIdToContinue);

    try {
      await generateMessage(convId, messageIdToContinue, onChunk, true);
    } catch (err) {
      console.error('Error continuing message');
    }
  };

  const saveConfig = (config: typeof CONFIG_DEFAULT) => {
    StorageUtils.setConfig(config);
    setConfig(config);
  };

  return (
    <AppContext.Provider
      value={{
        isGenerating,
        viewingChat,
        pendingMessages,
        sendMessage,
        stopGenerating,
        replaceMessageAndGenerate,
        continueMessageAndGenerate,
        canvasData,
        setCanvasData,
        config,
        saveConfig,
        showSettings,
        setShowSettings,
        serverProps,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
