// Plain-language lines for the tool calls of an agent trace: what one call did
// ("Read a file sheet_new.jpg"), what it is doing now ("Reading package.json")
// and what a run of calls did ("Ran 3 commands, read 2 files"). A call that
// has the argument the app picks (`description` by default) shows it as the
// model wrote it. Otherwise the app's resolvers phrase it, then llama-server's
// built-in tools, then a phrasing table, and any other tool reads as
// "Used <tool>" with its main argument.

import { extractToolArgString, fileNameFromPath, memoizeBounded, parsePartialJsonArgs } from "./args.js";
import {
  type ResolvedToolCall,
  type ToolArgField,
  type ToolCallDescription,
  type ToolCallInput,
  type ToolCallPhrasing,
  type ToolCallResolver,
  ToolCallSummaryKind,
  type TraceStep,
  TraceStepType,
} from "./types.js";

/** Longest target shown after the verb before it is cut with "...". */
export const TOOL_CALL_TARGET_MAX_CHARS = 60;

/** Arguments of a short, one-line value read as fields instead of JSON. */
export const TOOL_CALL_SIMPLE_ARG_MAX_CHARS = 200;

/** The argument a call's line shows when the call has it: the model's own words for it. */
export const TOOL_CALL_SHOWN_ARG_DEFAULT = "description";

/** Live title of a run of calls while the model thinks inside it. */
export const TOOL_CALL_THINKING_LABEL = "Thinking";

/** Arguments a tool without a phrasing is described by; the first one present wins. */
export const TOOL_CALL_MAIN_ARG_KEYS: readonly string[] = [
  "query",
  "q",
  "path",
  "file_path",
  "filePath",
  "url",
  "name",
  "command",
  "pattern",
  "prompt",
  "text",
  "id",
];

/** Argument names a path can come under. */
export const TOOL_ARG_PATH_KEYS: readonly string[] = ["path", "file_path", "filePath"];

function counted(n: number, singular: string, plural: string): string {
  return `${n} ${n === 1 ? singular : plural}`;
}

/** The group summary's clause per kind, in lower case; the summary capitalizes the first. */
export const TOOL_CALL_SUMMARY: Readonly<Record<ToolCallSummaryKind, (n: number) => string>> = {
  [ToolCallSummaryKind.COMMAND]: (n) => `ran ${counted(n, "command", "commands")}`,
  [ToolCallSummaryKind.EDIT]: (n) => `edited ${counted(n, "file", "files")}`,
  [ToolCallSummaryKind.LOAD_TOOLS]: (n) => `loaded ${counted(n, "tool", "tools")}`,
  [ToolCallSummaryKind.OTHER]: (n) => `used ${counted(n, "tool", "tools")}`,
  [ToolCallSummaryKind.READ]: (n) => `read ${counted(n, "file", "files")}`,
  [ToolCallSummaryKind.SCRIPT]: (n) => `ran ${counted(n, "script", "scripts")}`,
  [ToolCallSummaryKind.SEARCH]: (n) => (n === 1 ? "searched once" : `searched ${n} times`),
  [ToolCallSummaryKind.VIEW]: (n) => `viewed ${counted(n, "media file", "media files")}`,
  [ToolCallSummaryKind.WRITE]: (n) => `wrote ${counted(n, "file", "files")}`,
};

/** How llama-server's built-in tools read (`--tools`). */
export const LLAMA_SERVER_TOOL_PHRASING: Readonly<Record<string, ToolCallPhrasing>> = {
  edit_file: { object: "a file", past: "Edited", present: "Editing", summary: ToolCallSummaryKind.EDIT },
  exec_shell_command: {
    object: "a command",
    past: "Ran",
    present: "Running",
    summary: ToolCallSummaryKind.COMMAND,
  },
  file_glob_search: {
    object: "files",
    past: "Listed",
    present: "Listing",
    quoteTarget: true,
    summary: ToolCallSummaryKind.SEARCH,
    targetObject: "files matching",
  },
  get_info: { object: "runtime info", past: "Checked", present: "Checking", summary: ToolCallSummaryKind.OTHER },
  grep_search: {
    object: "files",
    past: "Searched",
    present: "Searching",
    quoteTarget: true,
    summary: ToolCallSummaryKind.SEARCH,
    targetObject: "for",
  },
  read_file: { object: "a file", past: "Read", present: "Reading", summary: ToolCallSummaryKind.READ },
  write_file: { object: "a file", past: "Wrote", present: "Writing", summary: ToolCallSummaryKind.WRITE },
};

function stringArg(args: Record<string, unknown> | null, key: string): string | undefined {
  const value = args?.[key];

  return typeof value === "string" && value.trim() ? value : undefined;
}

/** The path of a call, read without parsing arguments that may hold a whole file. */
export function toolCallPath(call: ToolCallInput): string | undefined {
  return extractToolArgString(call.rawArgs, TOOL_ARG_PATH_KEYS) ?? stringArg(call.args(), "path");
}

/** Phrases llama-server's built-in tools, with what each acted on. */
export const llamaServerToolResolver: ToolCallResolver = (call) => {
  const phrasing = LLAMA_SERVER_TOOL_PHRASING[call.name];

  if (!phrasing) return undefined;

  switch (call.name) {
    case "read_file":
    case "write_file":
    case "edit_file":
      return { phrasing, target: fileNameFromPath(toolCallPath(call)) };

    case "exec_shell_command":
      // the targeted read keeps the command in the line while a later
      // argument still streams and the JSON does not parse
      return {
        phrasing,
        target: stringArg(call.args(), "command") ?? extractToolArgString(call.rawArgs, ["command"]),
      };

    case "grep_search":
      return { phrasing, target: stringArg(call.args(), "pattern") };

    case "file_glob_search": {
      const include = stringArg(call.args(), "include");

      return { phrasing, target: include && include !== "**" ? include : undefined };
    }
  }

  return { phrasing };
};

function shorten(text: string): string {
  const oneLine = text.replace(/\s+/g, " ").trim();

  return oneLine.length > TOOL_CALL_TARGET_MAX_CHARS
    ? `${oneLine.slice(0, TOOL_CALL_TARGET_MAX_CHARS - 3)}...`
    : oneLine;
}

function genericMainArg(args: Record<string, unknown> | null): string | undefined {
  if (!args) return undefined;

  for (const key of TOOL_CALL_MAIN_ARG_KEYS) {
    const value = stringArg(args, key);

    if (value) return value;
  }

  for (const value of Object.values(args)) {
    if (typeof value === "number") return String(value);

    if (typeof value === "string" && value.trim() && value.length <= TOOL_CALL_TARGET_MAX_CHARS) {
      return value;
    }
  }

  return undefined;
}

function genericPhrasing(toolName: string): ToolCallPhrasing {
  return {
    object: toolName,
    past: "Used",
    present: "Using",
    summary: ToolCallSummaryKind.OTHER,
    targetObject: toolName,
  };
}

/** Results each cache below holds. */
const CACHE_MAX_SIZE = 256;
/** Arguments longer than this (whole files) are not worth holding as cache keys. */
const CACHE_MAX_ARGS_CHARS = 2000;

/** An argument name an app can pick: a plain key, so it is safe in a pattern. */
const SHOWN_ARG_PATTERN = /^[\w-]+$/;

/** The value of the argument a call's line shows, when the call has it. */
function shownArgValue(toolArgs: string, shownArg: string | null): string | undefined {
  if (!shownArg || !toolArgs || !SHOWN_ARG_PATTERN.test(shownArg)) return undefined;

  const value =
    toolArgs.length <= CACHE_MAX_ARGS_CHARS
      ? stringArg(parsePartialJsonArgs(toolArgs), shownArg)
      : extractToolArgString(toolArgs, [shownArg]);

  return value?.trim() || undefined;
}

/** One line of plain text for a description, e.g. `Searched for 'issue'`. */
export function formatToolCallLine(description: ToolCallDescription): string {
  if (!description.target) return description.label;

  const target = description.quoted ? `'${description.target}'` : description.target;

  return `${description.label} ${target}`;
}

export function isToolCallStep(step: TraceStep): boolean {
  return (
    step.type === TraceStepType.TOOL_CALL ||
    step.type === TraceStepType.TOOL_CALL_PENDING ||
    step.type === TraceStepType.TOOL_CALL_STREAMING
  );
}

export function isReasoningStep(step: TraceStep): boolean {
  return step.type === TraceStepType.REASONING || step.type === TraceStepType.REASONING_PENDING;
}

/** True while the call has no result yet and the reply is still streaming. */
export function isToolCallRunning(step: TraceStep, isStreaming: boolean): boolean {
  return (
    isStreaming &&
    (step.type === TraceStepType.TOOL_CALL_PENDING || step.type === TraceStepType.TOOL_CALL_STREAMING)
  );
}

export interface TracePhraserOptions {
  /** The app's own tools, tried in order before llama-server's built-in ones. */
  resolvers?: readonly ToolCallResolver[];
  /** Phrasing for tools whose line names no target, by tool name. */
  phrasings?: Readonly<Record<string, ToolCallPhrasing>>;
}

export interface TracePhraser {
  /**
   * One tool call as a short sentence, in the past tense once it is done and
   * the present tense while `running`. `shownArg` names the argument shown
   * instead when the call has it (null for none).
   */
  describeToolCall(
    toolName: string | undefined,
    toolArgs: string | undefined,
    running?: boolean,
    shownArg?: string | null,
  ): ToolCallDescription;
  /** A run of calls in one line, e.g. "Ran 3 commands, read 2 files". */
  summarizeToolCalls(steps: readonly TraceStep[]): string;
  /**
   * What a run of calls is doing now ("Running a command", "Thinking"): the
   * model thinking at its end, or the last call still running; null when neither.
   */
  describeToolActivity(
    steps: readonly TraceStep[],
    isStreaming: boolean,
    executingToolCallId?: string | null,
    shownArgFor?: (toolName: string | undefined) => string | null,
  ): string | null;
}

/** Phrasing for an app's traces: its own tools first, then llama-server's, then any tool. */
export function createTracePhraser(options: TracePhraserOptions = {}): TracePhraser {
  const resolvers = [...(options.resolvers ?? []), llamaServerToolResolver];
  const phrasings = options.phrasings ?? {};

  function resolveCall(name: string, rawArgs: string): ResolvedToolCall {
    let parsed: Record<string, unknown> | null | undefined;

    const call: ToolCallInput = {
      args: () => (parsed === undefined ? (parsed = rawArgs ? parsePartialJsonArgs(rawArgs) : null) : parsed),
      name,
      rawArgs,
    };

    for (const resolve of resolvers) {
      const resolved = resolve(call);

      if (resolved) return resolved;
    }

    const known = phrasings[name];

    if (known) return { phrasing: known };

    return { phrasing: genericPhrasing(name), target: genericMainArg(call.args()) };
  }

  function buildDescription(
    name: string,
    args: string,
    running: boolean,
    shownArg: string | null,
  ): ToolCallDescription {
    const { count = 1, phrasing, target } = resolveCall(name, args);
    const shown = shownArgValue(args, shownArg);
    // an argument the line already shows as its target (a shell command) would only repeat it
    const described = shown && shown !== target?.trim() ? shown : undefined;

    if (described) {
      return {
        count,
        described: true,
        // as the model wrote it: rewriting its tense by rule mangles anything
        // that does not start with a plain verb
        label: shorten(described).replace(/[.\s]+$/, ""),
        quoted: false,
        summary: phrasing.summary,
      };
    }

    const verb = running ? phrasing.present : phrasing.past;
    const shortTarget = target ? shorten(target) : undefined;
    const object = shortTarget ? phrasing.targetObject : phrasing.object;

    return {
      count,
      label: object ? `${verb} ${object}` : verb,
      quoted: Boolean(phrasing.quoteTarget && shortTarget),
      summary: phrasing.summary,
      target: shortTarget,
    };
  }

  // A trace is rebuilt on every streamed chunk, so the same finished call is
  // described again and again: remember the answer for its arguments. Every
  // caller of one call reads the same frozen object.
  const describeCached = memoizeBounded(
    (name: string, args: string, running: boolean, shownArg: string) =>
      Object.freeze(buildDescription(name, args, running, shownArg || null)),
    (name, args, running, shownArg) =>
      args.length <= CACHE_MAX_ARGS_CHARS ? `${running ? 1 : 0}\0${shownArg}\0${name}\0${args}` : null,
    CACHE_MAX_SIZE,
  );

  const describeToolCall: TracePhraser["describeToolCall"] = (
    toolName,
    toolArgs,
    running = false,
    shownArg = TOOL_CALL_SHOWN_ARG_DEFAULT,
  ) => describeCached(toolName || "a tool", toolArgs ?? "", running, shownArg ?? "");

  return {
    describeToolActivity(steps, isStreaming, executingToolCallId, shownArgFor = () => TOOL_CALL_SHOWN_ARG_DEFAULT) {
      // the model thinking at the end of the run, between calls or before its answer
      if (isStreaming && steps.at(-1)?.type === TraceStepType.REASONING_PENDING) {
        return TOOL_CALL_THINKING_LABEL;
      }

      for (let i = steps.length - 1; i >= 0; i--) {
        const step = steps[i];
        const executing = step.toolCallId !== undefined && step.toolCallId === executingToolCallId;

        if (executing || isToolCallRunning(step, isStreaming)) {
          return formatToolCallLine(describeToolCall(step.toolName, step.toolArgs, true, shownArgFor(step.toolName)));
        }
      }

      return null;
    },

    describeToolCall,

    summarizeToolCalls(steps) {
      const counts = new Map<ToolCallSummaryKind, number>();

      for (const step of steps) {
        if (!isToolCallStep(step)) continue;

        const { count, summary } = describeToolCall(step.toolName, step.toolArgs);

        counts.set(summary, (counts.get(summary) ?? 0) + count);
      }

      // kinds keep the order they first appear in; other tools come last
      const other = counts.get(ToolCallSummaryKind.OTHER);

      if (other !== undefined) {
        counts.delete(ToolCallSummaryKind.OTHER);
        counts.set(ToolCallSummaryKind.OTHER, other);
      }

      const text = [...counts].map(([kind, n]) => TOOL_CALL_SUMMARY[kind](n)).join(", ");

      return text.charAt(0).toUpperCase() + text.slice(1);
    },
  };
}

/**
 * Tool arguments as labelled fields when every value is a short, one-line
 * scalar; null when they need the JSON view instead. Memoized: the row that
 * shows them re-renders on every chunk.
 */
export function simpleToolArgFields(toolArgs: string | undefined): ToolArgField[] | null {
  return toolArgs ? argFieldsCached(toolArgs) : null;
}

const argFieldsCached = memoizeBounded(
  buildSimpleToolArgFields,
  (toolArgs) => (toolArgs.length <= CACHE_MAX_ARGS_CHARS ? toolArgs : null),
  CACHE_MAX_SIZE,
);

function buildSimpleToolArgFields(toolArgs: string): ToolArgField[] | null {
  let parsed: unknown;

  try {
    parsed = JSON.parse(toolArgs);
  } catch {
    return null;
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;

  const entries = Object.entries(parsed as Record<string, unknown>);

  if (entries.length === 0) return null;

  const fields: ToolArgField[] = [];

  for (const [key, value] of entries) {
    if (value !== null && typeof value === "object") return null;

    const text = String(value);

    if (text.length > TOOL_CALL_SIMPLE_ARG_MAX_CHARS || text.includes("\n")) return null;

    fields.push({ label: humanizeArgKey(key), value: text });
  }

  return fields;
}

/** `max_results` / `maxResults` -> `Max results`. */
export function humanizeArgKey(key: string): string {
  const words = key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim()
    .toLowerCase();

  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * Split step positions into single items and runs of tool work, e.g.
 * [0, [1, 2, 3], 4]. A run is the calls and all the reasoning around them, up
 * to the next text, so the answer follows the work as one row: it is grouped
 * once it holds `minCalls` calls, or one call with reasoning beside it.
 * Reasoning with no call in reach, and text, stay apart.
 */
export function groupToolCallRuns(
  steps: readonly TraceStep[],
  indices: readonly number[],
  minCalls: number,
): (number | number[])[] {
  const items: (number | number[])[] = [];

  let run: number[] = [];
  let calls = 0;
  // reasoning seen since the last call: it ends the run if text follows, and
  // leads into it if another call does
  let thinking: number[] = [];

  const flush = () => {
    const whole = [...run, ...thinking];

    if (calls >= minCalls || (calls > 0 && whole.length > calls)) items.push(whole);
    else items.push(...whole);

    run = [];
    thinking = [];
    calls = 0;
  };

  for (const index of indices) {
    const step = steps[index];

    if (isToolCallStep(step)) {
      run.push(...thinking, index);
      thinking = [];
      calls++;
    } else if (isReasoningStep(step)) {
      thinking.push(index);
    } else {
      flush();
      items.push(index);
    }
  }

  flush();

  return items;
}
