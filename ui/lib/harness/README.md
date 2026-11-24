# @ggml-org/llama-harness

Llama Harness, the agentic loop of Llama Chat, as a package: it streams a turn, runs the tools the model called, gives it the results, and repeats until the model answers without a tool call. It is headless TypeScript with no UI, no DOM and no dependencies, so the same loop runs in a browser, in Node and in a worker. It lives in llama.cpp at `ui/lib/harness`.

It talks to llama-server's chat completions endpoint (`/v1/chat/completions`) and reads the timings and prompt progress llama-server adds.

## Use

```ts
import { createHarness, createOpenAIClient, type Tool } from "@ggml-org/llama-harness";

const harness = createHarness(createOpenAIClient({ baseUrl: "http://localhost:8080" }));

const getTime: Tool = {
  definition: {
    type: "function",
    function: {
      name: "get_time",
      description: "The current time",
      parameters: { type: "object", properties: {} },
    },
  },
  execute: async () => ({ content: new Date().toISOString() }),
};

for await (const event of harness.run([{ role: "user", content: "What time is it?" }], [getTime])) {
  if (event.type === "text") process.stdout.write(event.delta);
}
```

## The contract

```ts
interface Harness {
  run(messages, tools, signal?, options?): AsyncIterable<HarnessEvent>;
}
```

- **Messages and tools** use the OpenAI shapes. A tool is its definition plus an `execute` function.
- **Events:** `turn_start`, `text`, `reasoning`, `tool_call_delta`, `timings`, `model`, `completion_id`, `message` (a turn's assistant message), `tool_call`, `tool_update` (a streaming tool's output so far), `tool_result`, `turn_end`, and last `done` or `error`. `run` never throws.
- **The loop waits for you.** It does not go on until you ask for the next event, so you can store each message before the next step runs.
- **It keeps nothing.** No conversation state, no storage: the app owns the history and passes it in.
- **Stopping:** `signal` stops at once. `options.interrupted` ends the run at the next safe point, e.g. when the user sent a new message. `options.maxTurns` with `options.onTurnLimit` asks before it goes past a turn limit.

An app can put another loop behind the same `Harness` interface. What sits above the loop is the same for every loop: the tools, and whatever an app wraps around `execute`, such as a permission prompt.

## Traces

A run's steps also read as an agent trace: what each tool call did, in plain words. That part is headless too, so any app on the harness can show the same lines.

```ts
import { createTracePhraser, formatToolCallLine, groupToolCallRuns } from "@ggml-org/llama-harness";

const { describeToolCall, summarizeToolCalls, describeToolActivity } = createTracePhraser();

formatToolCallLine(describeToolCall("read_file", '{"path":"src/app.ts"}'));  // "Read app.ts"
summarizeToolCalls(steps);                                                    // "Ran 3 commands, read 2 files"
describeToolActivity(steps, true);                                            // "Running npm test", or "Thinking"
```

- **Steps** are anything with a `type` from `TraceStepType` (text, reasoning, or a tool call that is done, pending or still streaming) plus the call's name and arguments.
- **Phrasing** knows llama-server's built-in tools. An app adds its own tools with `resolvers` (a function per tool or family of tools) and `phrasings` (a table by name); any other tool reads as "Used <tool>" with its main argument. A call whose arguments carry `description` shows the model's own words instead.
- **Streaming:** arguments are read while they stream (`parsePartialJsonArgs`), and arguments that hold whole files are read for their path only (`extractToolArgString`).
- **Grouping:** `groupToolCallRuns` folds the calls and the reasoning around them into one row, up to the next text.

## What the app keeps

Llama Chat (`ui/apps/chat`) is the first app on it. Its agentic store gives the harness:

- **An inference client:** its own chat service, which knows the servers, the settings and how to resume a stream.
- **The tools:** built-in tools, llama-server tools and MCP tools, each behind the permission gate.
- **Storage:** it turns the events into messages in its database.

## Develop

```sh
npm ci
npm run check     # type check
npm test          # unit tests
npm run prepack   # build dist/ and lint the package
```

Llama Chat does not need a build of this package: it compiles `src/` directly, through the `@ggml-org/llama-harness` alias in its `vite.config.ts`.
