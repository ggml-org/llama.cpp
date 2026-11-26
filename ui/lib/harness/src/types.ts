// Messages and tools use the OpenAI chat-completions shapes, so they go to the endpoint as they are.

export interface TextPart {
  type: "text";
  text: string;
}

export interface ImagePart {
  type: "image_url";
  image_url: { url: string };
}

export interface AudioPart {
  type: "input_audio";
  input_audio: { data: string; format: string };
}

export type ContentPart = TextPart | ImagePart | AudioPart;

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

/** A tool call as it streams in: any field can still be missing. */
export interface PartialToolCall {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string };
}

export type Message =
  | { role: "system" | "user"; content: string | ContentPart[] }
  | {
      role: "assistant";
      content?: string | ContentPart[];
      reasoning_content?: string;
      tool_calls?: ToolCall[];
    }
  | { role: "tool"; tool_call_id: string; content: string | ContentPart[] };

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface ToolContext {
  signal?: AbortSignal;
  /** Report the output so far of a tool that streams. Each call replaces the previous content. */
  update(content: string): void;
}

export interface ToolResult {
  /** Text the model reads. */
  content: string;
  /** Media the model reads after the text, e.g. an image the tool produced. */
  parts?: ContentPart[];
  isError?: boolean;
  /** Set it when the tool times itself, e.g. to leave out a wait for approval. Default: the time execute took. */
  durationMs?: number;
  /** App data the loop does not read. It comes back on the tool_result event. */
  details?: unknown;
}

export interface Tool {
  definition: ToolDefinition;
  /** Run the call. Reject with an AbortError to end the run, with any other error to report it to the model. */
  execute(call: ToolCall, context: ToolContext): Promise<ToolResult>;
}

/** Token counts and times of one request, as llama-server reports them. */
export interface LlmTimings {
  cache_n?: number;
  predicted_ms?: number;
  predicted_n?: number;
  prompt_ms?: number;
  prompt_n?: number;
}

export interface LlmTotals {
  predicted_ms: number;
  predicted_n: number;
  prompt_ms: number;
  prompt_n: number;
}

export interface ToolCallTiming {
  /** The call's id, to match the timing to its call. */
  id: string;
  name: string;
  duration_ms: number;
  success: boolean;
  /** Wall-clock start and end (ms since epoch), waits for approval included. */
  startedAt: number;
  endedAt: number;
}

export interface TurnStats {
  turn: number;
  llm: LlmTotals;
  toolCalls: ToolCallTiming[];
  toolsMs: number;
}

/** Totals of a run: every turn's requests and tool calls. */
export interface RunTimings {
  turns: number;
  toolCallsCount: number;
  toolsMs: number;
  toolCalls: ToolCallTiming[];
  perTurn: TurnStats[];
  llm: LlmTotals;
}

/** The last request's timings, with the run's totals once a tool has run. */
export interface Timings extends LlmTimings {
  agentic?: RunTimings;
}

export interface PromptProgress {
  cache: number;
  processed: number;
  time_ms: number;
  total: number;
}

export interface InferenceRequest {
  messages: Message[];
  tools?: ToolDefinition[];
  signal?: AbortSignal;
}

export type InferenceEvent =
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  /** Every tool call of the response so far, not only the new part. */
  | { type: "tool_calls"; toolCalls: PartialToolCall[] }
  | { type: "timings"; timings?: LlmTimings; progress?: PromptProgress }
  | { type: "model"; model: string }
  | { type: "completion_id"; id: string };

/**
 * One streamed chat completion. The loop needs nothing else from llama-server.
 * Timings, model and completion id are optional extras.
 */
export interface InferenceClient {
  stream(request: InferenceRequest): AsyncIterable<InferenceEvent>;
}

export type DoneReason =
  /** The model answered without calling a tool. */
  | "complete"
  | "aborted"
  /** `interrupted` returned true. */
  | "interrupted"
  /** The turn limit was reached and `onTurnLimit` did not grant more turns. */
  | "turn_limit";

export type HarnessEvent =
  /** `turn` starts at 1, and again at 1 once `onTurnLimit` grants more turns. */
  | { type: "turn_start"; turn: number }
  | { type: "text"; delta: string }
  | { type: "reasoning"; delta: string }
  | { type: "tool_call_delta"; toolCalls: PartialToolCall[] }
  /** `llm` is the run's totals including the request in flight. */
  | { type: "timings"; timings?: LlmTimings; progress?: PromptProgress; llm?: LlmTotals }
  | { type: "model"; model: string }
  | { type: "completion_id"; id: string }
  /** The assistant message of a turn. `final` is true when the run ends after it and none of its tool calls runs. */
  | {
      type: "message";
      content: string;
      reasoning?: string;
      toolCalls?: ToolCall[];
      timings?: Timings;
      final: boolean;
    }
  | { type: "tool_call"; toolCall: ToolCall }
  | { type: "tool_update"; toolCall: ToolCall; content: string }
  /** `skipped` is true when the call did not run because the run was interrupted. */
  | { type: "tool_result"; toolCall: ToolCall; result: ToolResult; skipped: boolean }
  | { type: "turn_end"; timings: Timings }
  | { type: "done"; reason: DoneReason; timings?: Timings }
  | { type: "error"; error: Error; timings?: Timings };

export interface RunOptions {
  /** Turns before `onTurnLimit` is asked. Default: no limit. */
  maxTurns?: number;
  /** Resolve true to run `maxTurns` more turns. Default: the run ends. */
  onTurnLimit?: (signal?: AbortSignal) => Promise<boolean>;
  /**
   * Return true to end the run at the next safe point, e.g. when the user sent a new message. Tool calls that did
   * not run yet get a result that says so. Use `signal` to stop at once.
   */
  interrupted?: () => boolean;
}

/**
 * The contract an app runs a loop through. `run` never throws: it ends with a `done` or an `error` event. It waits
 * for the consumer between events, so the consumer can store each one before the loop goes on.
 */
export interface Harness {
  run(
    messages: Message[],
    tools: Tool[],
    signal?: AbortSignal,
    options?: RunOptions,
  ): AsyncIterable<HarnessEvent>;
}
