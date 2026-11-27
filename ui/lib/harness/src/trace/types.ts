/**
 * What a step of an agent trace is: the model's text, its reasoning, or a tool
 * call, finished or still in flight. Apps keep their own step type; anything
 * with these values in `type` reads as a trace step.
 */
export const TraceStepType = {
  REASONING: "reasoning",
  REASONING_PENDING: "reasoning_pending",
  TEXT: "text",
  TOOL_CALL: "tool_call",
  TOOL_CALL_PENDING: "tool_call_pending",
  TOOL_CALL_STREAMING: "tool_call_streaming",
} as const;

export type TraceStepType = (typeof TraceStepType)[keyof typeof TraceStepType];

/** One step of an agent trace, as much of it as the phrasing reads. */
export interface TraceStep {
  /** A TraceStepType value. */
  type: string;
  toolCallId?: string;
  toolName?: string;
  /** The call's arguments as streamed: JSON, possibly cut short. */
  toolArgs?: string;
}

/** Which clause a call counts under in a group summary ("ran 3 commands"). */
export const ToolCallSummaryKind = {
  COMMAND: "command",
  EDIT: "edit",
  LOAD_TOOLS: "load_tools",
  OTHER: "other",
  READ: "read",
  SCRIPT: "script",
  SEARCH: "search",
  VIEW: "view",
  WRITE: "write",
} as const;

export type ToolCallSummaryKind = (typeof ToolCallSummaryKind)[keyof typeof ToolCallSummaryKind];

/** How a tool call reads in a trace. */
export interface ToolCallPhrasing {
  /** Verb once the call is done, e.g. "Read". */
  past: string;
  /** Verb while the call runs, e.g. "Reading". */
  present: string;
  /** Words after the verb when the call has no target, e.g. "a file". */
  object: string;
  /** Words between the verb and the target, e.g. "for". */
  targetObject?: string;
  /** Show the target in quotes, for search terms. */
  quoteTarget?: boolean;
  summary: ToolCallSummaryKind;
}

/** A tool call as one plain sentence: label, then an optional target. */
export interface ToolCallDescription {
  /** Verb and object, e.g. "Read a file" or "Searched for". */
  label: string;
  /** What the call acted on, e.g. a file name or a search term. */
  target?: string;
  quoted: boolean;
  summary: ToolCallSummaryKind;
  /** How many things the call counts as in the group summary. */
  count: number;
  /** True when the label is the model's own description of the call. */
  described?: boolean;
}

/** A tool call as a resolver sees it. */
export interface ToolCallInput {
  name: string;
  /** The arguments as streamed. Whole files for write tools: read them with extractToolArgString. */
  rawArgs: string;
  /** The arguments parsed leniently, cut-off JSON included; parsed on first call. */
  args(): Record<string, unknown> | null;
}

/** How one call reads: its phrasing, what it acted on, and how many things it counts as. */
export interface ResolvedToolCall {
  phrasing: ToolCallPhrasing;
  target?: string;
  count?: number;
}

/** Phrases the calls of one tool, or of a family of tools; undefined passes the call on. */
export type ToolCallResolver = (call: ToolCallInput) => ResolvedToolCall | undefined;

/** A tool argument shown as a labelled field. */
export interface ToolArgField {
  label: string;
  value: string;
}
