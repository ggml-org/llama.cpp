import { isAbortError } from "./abort.js";
import type {
  DoneReason,
  Harness,
  HarnessEvent,
  InferenceClient,
  LlmTimings,
  LlmTotals,
  Message,
  PartialToolCall,
  RunOptions,
  RunTimings,
  Timings,
  Tool,
  ToolCall,
  ToolResult,
  TurnStats,
} from "./types.js";

/** The result of a tool call that did not run because the run was interrupted. */
export const INTERRUPTED_TOOL_RESULT = "Tool execution was interrupted by a new user message.";

function emptyTotals(): LlmTotals {
  return { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 };
}

function addTimings(totals: LlmTotals, timings: LlmTimings): LlmTotals {
  return {
    predicted_ms: totals.predicted_ms + (timings.predicted_ms ?? 0),
    predicted_n: totals.predicted_n + (timings.predicted_n ?? 0),
    prompt_ms: totals.prompt_ms + (timings.prompt_ms ?? 0),
    prompt_n: totals.prompt_n + (timings.prompt_n ?? 0),
  };
}

function toToolCalls(partial: PartialToolCall[]): ToolCall[] {
  return partial.map((call, index) => ({
    function: {
      arguments: call?.function?.arguments ?? "",
      name: call?.function?.name ?? "",
    },
    id: call?.id ?? `tool_${index}`,
    type: "function",
  }));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * The default loop: stream a turn, run the tools the model called, give it the results, and repeat until it answers
 * without a tool call. It keeps no state between runs and stores nothing: the app owns the conversation.
 */
export function createHarness(client: InferenceClient): Harness {
  return {
    run: (messages, tools, signal, options) => run(client, messages, tools, signal, options),
  };
}

async function* run(
  client: InferenceClient,
  messages: Message[],
  tools: Tool[],
  signal?: AbortSignal,
  options: RunOptions = {},
): AsyncGenerator<HarnessEvent> {
  const { interrupted = () => false, maxTurns = Infinity, onTurnLimit } = options;
  const history = [...messages];
  const definitions = tools.map((tool) => tool.definition);
  const toolsByName = new Map(tools.map((tool) => [tool.definition.function.name, tool]));
  const totals: RunTimings = {
    llm: emptyTotals(),
    perTurn: [],
    toolCalls: [],
    toolCallsCount: 0,
    toolsMs: 0,
    turns: 0,
  };

  let lastTimings: LlmTimings | undefined;
  let turn = 0;

  const runTimings = (): Timings | undefined => {
    if (totals.toolCallsCount === 0) return lastTimings;

    return {
      agentic: totals,
      cache_n: lastTimings?.cache_n,
      predicted_ms: lastTimings?.predicted_ms,
      predicted_n: lastTimings?.predicted_n,
      prompt_ms: lastTimings?.prompt_ms,
      prompt_n: lastTimings?.prompt_n,
    };
  };
  const done = (reason: DoneReason): HarnessEvent => ({
    reason,
    timings: runTimings(),
    type: "done",
  });

  while (true) {
    if (turn >= maxTurns) {
      const more = onTurnLimit ? await onTurnLimit(signal) : false;

      if (!more || signal?.aborted) {
        yield done(signal?.aborted ? "aborted" : "turn_limit");

        return;
      }

      turn = 0;
    }

    totals.turns = turn + 1;

    if (signal?.aborted) {
      yield done("aborted");

      return;
    }

    yield { turn: turn + 1, type: "turn_start" };

    let content = "";
    let reasoning = "";
    let partialCalls: PartialToolCall[] = [];
    let turnTimings: LlmTimings | undefined;

    const stats: TurnStats = { llm: emptyTotals(), toolCalls: [], toolsMs: 0, turn: turn + 1 };
    // the turn's message when the run ends after it: what streamed so far, with the run's timings
    const lastMessage = (toolCalls?: ToolCall[]): HarnessEvent => ({
      content,
      final: true,
      reasoning: reasoning || undefined,
      timings: runTimings(),
      toolCalls,
      type: "message",
    });

    try {
      const stream = client.stream({
        messages: history,
        signal,
        tools: definitions.length > 0 ? definitions : undefined,
      });

      for await (const event of stream) {
        if (event.type === "text") {
          content += event.delta;
          yield event;
        } else if (event.type === "reasoning") {
          reasoning += event.delta;
          yield event;
        } else if (event.type === "tool_calls") {
          partialCalls = event.toolCalls;
          yield { toolCalls: partialCalls, type: "tool_call_delta" };
        } else if (event.type === "timings") {
          if (event.timings) {
            lastTimings = event.timings;
            turnTimings = event.timings;
          }

          yield { ...event, llm: event.timings && addTimings(totals.llm, event.timings) };
        } else {
          yield event;
        }
      }

      if (turnTimings) {
        totals.llm = addTimings(totals.llm, turnTimings);
        stats.llm = addTimings(stats.llm, turnTimings);
      }
    } catch (error) {
      // keep the partial output in both cases
      yield lastMessage();

      if (signal?.aborted) {
        yield done("aborted");
      } else {
        yield {
          error: error instanceof Error ? error : new Error("LLM stream error"),
          timings: runTimings(),
          type: "error",
        };
      }

      return;
    }

    // a client can swallow the abort and end the stream normally: do not take a cut tool call for a complete one
    if (signal?.aborted) {
      yield lastMessage();
      yield done("aborted");

      return;
    }

    if (interrupted()) {
      yield lastMessage(partialCalls.length > 0 ? toToolCalls(partialCalls) : undefined);
      yield done("interrupted");

      return;
    }

    if (partialCalls.length === 0) {
      totals.perTurn.push(stats);

      const timings = runTimings();

      yield lastMessage();

      if (timings) yield { timings, type: "turn_end" };

      yield done("complete");

      return;
    }

    const toolCalls = toToolCalls(partialCalls);

    yield {
      content,
      final: false,
      reasoning: reasoning || undefined,
      timings: turnTimings,
      toolCalls,
      type: "message",
    };

    history.push({
      content: content || undefined,
      reasoning_content: reasoning || undefined,
      role: "assistant",
      tool_calls: toolCalls,
    });

    for (let i = 0; i < toolCalls.length; i++) {
      const toolCall = toolCalls[i];

      if (signal?.aborted) {
        yield done("aborted");

        return;
      }

      if (interrupted()) {
        for (const skippedCall of toolCalls.slice(i)) {
          yield {
            result: { content: INTERRUPTED_TOOL_RESULT },
            skipped: true,
            toolCall: skippedCall,
            type: "tool_result",
          };

          history.push({
            content: INTERRUPTED_TOOL_RESULT,
            role: "tool",
            tool_call_id: skippedCall.id,
          });
        }

        break;
      }

      yield { toolCall, type: "tool_call" };

      const tool = toolsByName.get(toolCall.function.name);
      const startTime = performance.now();

      let result: ToolResult;

      if (!tool) {
        result = { content: `Error: Unknown tool: ${toolCall.function.name}`, isError: true };
      } else {
        // only the latest update matters, so a slow consumer skips the ones in between
        let pendingUpdate: string | undefined;
        let settled = false;
        let wake: (() => void) | undefined;

        const notify = () => {
          wake?.();
          wake = undefined;
        };
        const execution = Promise.resolve()
          .then(() =>
            tool.execute(toolCall, {
              signal,
              update: (updated) => {
                pendingUpdate = updated;
                notify();
              },
            }),
          )
          .finally(() => {
            settled = true;
            notify();
          });

        // the rejection is handled below, after the updates are drained
        execution.catch(() => {});

        while (true) {
          if (pendingUpdate !== undefined) {
            const updated = pendingUpdate;

            pendingUpdate = undefined;
            yield { content: updated, toolCall, type: "tool_update" };
          } else if (settled) {
            break;
          } else {
            await new Promise<void>((resolve) => (wake = resolve));
          }
        }

        try {
          result = await execution;
        } catch (error) {
          if (isAbortError(error)) {
            yield done("aborted");

            return;
          }

          result = { content: `Error: ${errorMessage(error)}`, isError: true };
        }
      }

      const durationMs = Math.round(result.durationMs ?? performance.now() - startTime);
      const timing = {
        duration_ms: durationMs,
        name: toolCall.function.name,
        success: !result.isError,
      };

      totals.toolCalls.push(timing);
      totals.toolCallsCount++;
      totals.toolsMs += durationMs;
      stats.toolCalls.push(timing);
      stats.toolsMs += durationMs;

      if (signal?.aborted) {
        yield done("aborted");

        return;
      }

      yield { result, skipped: false, toolCall, type: "tool_result" };

      history.push({
        content: result.parts?.length
          ? [{ text: result.content, type: "text" }, ...result.parts]
          : result.content,
        role: "tool",
        tool_call_id: toolCall.id,
      });
    }

    if (stats.toolCalls.length > 0) {
      totals.perTurn.push(stats);

      const timings = runTimings();

      if (timings) yield { timings, type: "turn_end" };
    }

    if (interrupted()) {
      yield done("interrupted");

      return;
    }

    turn++;
  }
}
