import { useMemo, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import { Message, PendingMessage } from '../utils/types';
import { classNames } from '../utils/misc';
import MarkdownDisplay, { CopyButton } from './MarkdownDisplay';
import { ToolCallArgsDisplay } from './tool_calling/ToolCallArgsDisplay';
import { ToolCallResultDisplay } from './tool_calling/ToolCallResultDisplay';
import {
  ArrowPathIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  PencilSquareIcon,
} from '@heroicons/react/24/outline';
import ChatInputExtraContextItem from './ChatInputExtraContextItem';
import { BtnWithTooltips } from '../utils/common';

interface SplitMessage {
  content: PendingMessage['content'];
  thought?: string;
  isThinking?: boolean;
}

// Helper function to extract thoughts from message content
function extractThoughts(content: string | null, role: string): SplitMessage {
  if (content === null || (role !== 'assistant' && role !== 'tool')) {
    return { content };
  }

  let actualContent = '';
  let thought = '';
  let isThinking = false;
  let thinkSplit = content.split('<think>', 2);
  actualContent += thinkSplit[0];

  while (thinkSplit[1] !== undefined) {
    thinkSplit = thinkSplit[1].split('</think>', 2);
    thought += thinkSplit[0];
    isThinking = true;
    if (thinkSplit[1] !== undefined) {
      isThinking = false;
      thinkSplit = thinkSplit[1].split('<think>', 2);
      actualContent += thinkSplit[0];
    }
  }

  return { content: actualContent, thought, isThinking };
}

// Helper component to render a single message part
function MessagePart({
  message,
  isPending,
  showThoughts = true,
  className = '',
  baseClassName = '',
  isMainMessage = false,
}: {
  message: Message | PendingMessage;
  isPending?: boolean;
  showThoughts?: boolean;
  className?: string;
  baseClassName?: string;
  isMainMessage?: boolean;
}) {
  const { config } = useAppContext();
  const { content, thought, isThinking } = extractThoughts(
    message.content,
    message.role
  );

  if (message.role === 'tool' && baseClassName) {
    return (
      <ToolCallResultDisplay
        content={content || ''}
        baseClassName={baseClassName}
      />
    );
  }

  return (
    <div className={className}>
      {showThoughts && thought && (
        <ThoughtProcess
          isThinking={!!isThinking && !!isPending}
          content={thought}
          open={config.showThoughtInProgress}
        />
      )}

      {message.role === 'tool' && content ? (
        <ToolCallResultDisplay content={content} />
      ) : (
        content &&
        content.trim() !== '' && (
          <MarkdownDisplay content={content} isGenerating={isPending} />
        )
      )}

      {message.tool_calls &&
        message.tool_calls.map((toolCall) => (
          <ToolCallArgsDisplay
            key={toolCall.id}
            toolCall={toolCall}
            {...(!isMainMessage && baseClassName ? { baseClassName } : {})}
          />
        ))}
    </div>
  );
}

export default function ChatMessage({
  msg,
  chainedParts,
  siblingLeafNodeIds,
  siblingCurrIdx,
  id,
  onRegenerateMessage,
  onEditMessage,
  onChangeSibling,
  isPending,
}: {
  msg: Message | PendingMessage;
  chainedParts?: (Message | PendingMessage)[];
  siblingLeafNodeIds: Message['id'][];
  siblingCurrIdx: number;
  id?: string;
  onRegenerateMessage(msg: Message): void;
  onEditMessage(msg: Message, content: string): void;
  onChangeSibling(sibling: Message['id']): void;
  isPending?: boolean;
}) {
  const { viewingChat, config } = useAppContext();
  const [editingContent, setEditingContent] = useState<string | null>(null);
  const timings = useMemo(
    () =>
      msg.timings
        ? {
            ...msg.timings,
            prompt_per_second:
              (msg.timings.prompt_n / msg.timings.prompt_ms) * 1000,
            predicted_per_second:
              (msg.timings.predicted_n / msg.timings.predicted_ms) * 1000,
          }
        : null,
    [msg.timings]
  );
  const nextSibling = siblingLeafNodeIds[siblingCurrIdx + 1];
  const prevSibling = siblingLeafNodeIds[siblingCurrIdx - 1];

  const mainSplitMessage = useMemo(
    () => extractThoughts(msg.content, msg.role),
    [msg.content, msg.role]
  );

  if (!viewingChat) return null;

  const toolCalls = msg.tool_calls ?? null;

  const hasContentInMainMsg =
    mainSplitMessage.content && mainSplitMessage.content.trim() !== '';
  const hasContentInChainedParts = chainedParts?.some((part) => {
    const splitPart = extractThoughts(part.content, part.role);
    return splitPart.content && splitPart.content.trim() !== '';
  });
  const entireTurnHasSomeDisplayableContent =
    hasContentInMainMsg || hasContentInChainedParts;
  const isUser = msg.role === 'user';

  return (
    <div
      className="group"
      id={id}
      role="group"
      aria-description={`Message from ${msg.role}`}
    >
      <div
        className={classNames({
          chat: true,
          'chat-start': !isUser,
          'chat-end': isUser,
        })}
      >
        {msg.extra && msg.extra.length > 0 && (
          <ChatInputExtraContextItem items={msg.extra} clickToShow />
        )}

        <div
          className={classNames({
            'chat-bubble markdown': true,
            'chat-bubble bg-transparent': !isUser,
          })}
        >
          {/* textarea for editing message */}
          {editingContent !== null && (
            <>
              <textarea
                dir="auto"
                className="textarea textarea-bordered bg-base-100 text-base-content max-w-2xl w-[calc(90vw-8em)] h-24"
                value={editingContent}
                onChange={(e) => setEditingContent(e.target.value)}
              ></textarea>
              <br />
              <button
                className="btn btn-ghost mt-2 mr-2"
                onClick={() => setEditingContent(null)}
              >
                Cancel
              </button>
              <button
                className="btn mt-2"
                onClick={() => {
                  if (msg.content !== null) {
                    setEditingContent(null);
                    onEditMessage(msg as Message, editingContent);
                  }
                }}
              >
                Submit
              </button>
            </>
          )}
          {/* not editing content, render message */}
          {editingContent === null && (
            <>
              {mainSplitMessage.content === null &&
              !toolCalls &&
              !chainedParts?.length ? (
                <>
                  {/* show loading dots for pending message */}
                  <span className="loading loading-dots loading-md"></span>
                </>
              ) : (
                <>
                  {/* render main message */}
                  <div dir="auto" tabIndex={0}>
                    <MessagePart
                      message={msg}
                      isPending={isPending}
                      showThoughts={true}
                      isMainMessage={true}
                    />
                  </div>

                  {/* render chained parts */}
                  {chainedParts?.map((part) => (
                    <MessagePart
                      key={part.id}
                      message={part}
                      isPending={isPending}
                      showThoughts={true}
                      className={part.role === 'assistant' ? 'mt-2' : ''}
                      baseClassName={
                        part.role === 'tool'
                          ? 'collapse bg-base-200 collapse-arrow mb-4 mt-2'
                          : ''
                      }
                    />
                  ))}
                </>
              )}
              {/* render timings if enabled */}
              {timings && config.showTokensPerSecond && (
                <div className="dropdown dropdown-hover dropdown-top mt-2">
                  <div
                    tabIndex={0}
                    role="button"
                    className="cursor-pointer font-semibold text-sm opacity-60"
                  >
                    Speed: {timings.predicted_per_second.toFixed(1)} t/s
                  </div>
                  <div className="dropdown-content bg-base-100 z-10 w-64 p-2 shadow mt-4">
                    <b>Prompt</b>
                    <br />- Tokens: {timings.prompt_n}
                    <br />- Time: {timings.prompt_ms} ms
                    <br />- Speed: {timings.prompt_per_second.toFixed(1)} t/s
                    <br />
                    <b>Generation</b>
                    <br />- Tokens: {timings.predicted_n}
                    <br />- Time: {timings.predicted_ms} ms
                    <br />- Speed: {timings.predicted_per_second.toFixed(1)} t/s
                    <br />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* actions for each message */}
      {(entireTurnHasSomeDisplayableContent || msg.role === 'user') && (
        <div
          className={classNames({
            'flex items-center gap-2 mx-4 mt-2 mb-2': true,
            'flex-row-reverse': msg.role === 'user',
          })}
        >
          {siblingLeafNodeIds && siblingLeafNodeIds.length > 1 && (
            <div
              className="flex gap-1 items-center opacity-60 text-sm"
              role="navigation"
              aria-description={`Message version ${siblingCurrIdx + 1} of ${siblingLeafNodeIds.length}`}
            >
              <button
                className={classNames({
                  'btn btn-sm btn-ghost p-1': true,
                  'opacity-20': !prevSibling,
                })}
                onClick={() => prevSibling && onChangeSibling(prevSibling)}
                aria-label="Previous message version"
              >
                <ChevronLeftIcon className="h-4 w-4" />
              </button>
              <span>
                {siblingCurrIdx + 1} / {siblingLeafNodeIds.length}
              </span>
              <button
                className={classNames({
                  'btn btn-sm btn-ghost p-1': true,
                  'opacity-20': !nextSibling,
                })}
                onClick={() => nextSibling && onChangeSibling(nextSibling)}
                aria-label="Next message version"
              >
                <ChevronRightIcon className="h-4 w-4" />
              </button>
            </div>
          )}
          {/* user message */}
          {msg.role === 'user' && (
            <BtnWithTooltips
              className="btn-mini w-8 h-8"
              onClick={() => setEditingContent(msg.content)}
              disabled={msg.content === null}
              tooltipsContent="Edit message"
            >
              <PencilSquareIcon className="h-4 w-4" />
            </BtnWithTooltips>
          )}
          {/* assistant message */}
          {msg.role === 'assistant' && (
            <>
              {!isPending && (
                <BtnWithTooltips
                  className="btn-mini w-8 h-8"
                  onClick={() => {
                    if (entireTurnHasSomeDisplayableContent) {
                      onRegenerateMessage(msg as Message);
                    }
                  }}
                  disabled={
                    !entireTurnHasSomeDisplayableContent || msg.content === null
                  }
                  tooltipsContent="Regenerate response"
                >
                  <ArrowPathIcon className="h-4 w-4" />
                </BtnWithTooltips>
              )}
            </>
          )}
          {entireTurnHasSomeDisplayableContent && (
            <CopyButton
              className="badge btn-mini show-on-hover mr-2"
              content={
                [msg, ...(chainedParts || [])]
                  .filter((p) => p.content)
                  .map((p) => {
                    if (p.role === 'user') {
                      return p.content;
                    } else {
                      return extractThoughts(p.content, p.role).content;
                    }
                  })
                  .join('\n\n') || ''
              }
            />
          )}
        </div>
      )}
    </div>
  );
}

function ThoughtProcess({
  isThinking,
  content,
  open,
}: {
  isThinking: boolean;
  content: string;
  open: boolean;
}) {
  return (
    <div
      role="button"
      aria-label="Toggle thought process display"
      tabIndex={0}
      className={classNames({
        'collapse bg-none': true,
      })}
    >
      <input type="checkbox" defaultChecked={open} />
      <div className="collapse-title px-0">
        <div className="btn rounded-xl">
          {isThinking ? (
            <span>
              <span
                className="loading loading-spinner loading-md mr-2"
                style={{ verticalAlign: 'middle' }}
              ></span>
              Thinking
            </span>
          ) : (
            <>Thought Process</>
          )}
        </div>
      </div>
      <div
        className="collapse-content text-base-content/70 text-sm p-1"
        tabIndex={0}
        aria-description="Thought process content"
      >
        <div className="border-l-2 border-base-content/20 pl-4 mb-4">
          <MarkdownDisplay content={content} />
        </div>
      </div>
    </div>
  );
}
