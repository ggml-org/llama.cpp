import { describe, expect, it } from "vitest";
import {
  createTracePhraser,
  extractToolArgString,
  formatToolCallLine,
  groupToolCallRuns,
  humanizeArgKey,
  parsePartialJsonArgs,
  simpleToolArgFields,
  ToolCallSummaryKind,
  type TraceStep,
  TraceStepType,
} from "../src/index.js";

function call(toolName: string, args: object | string, type: string = TraceStepType.TOOL_CALL): TraceStep {
  return { toolArgs: typeof args === "string" ? args : JSON.stringify(args), toolName, type };
}

const phraser = createTracePhraser();
const line = (name: string, args: object | string, running = false) =>
  formatToolCallLine(phraser.describeToolCall(name, typeof args === "string" ? args : JSON.stringify(args), running));

describe("parsePartialJsonArgs", () => {
  it("reads the keys of arguments cut off mid-stream", () => {
    expect(parsePartialJsonArgs('{"path": "src/a.ts", "content": "line')).toEqual({
      content: "line",
      path: "src/a.ts",
    });
    expect(parsePartialJsonArgs('{"items": [1, 2,')).toEqual({ items: [1, 2] });
    expect(parsePartialJsonArgs("[1, 2]")).toBeNull();
  });

  it("reads one field of a whole-file argument without parsing it", () => {
    expect(extractToolArgString('{"content": "x\\"y", "path": "a\\\\b.ts"}', ["file_path", "path"])).toBe("a\\b.ts");
  });
});

describe("llama-server tools", () => {
  it("say what each call acted on, in the past or present tense", () => {
    expect(line("read_file", { path: "/repo/src/app.ts" })).toBe("Read app.ts");
    expect(line("read_file", { path: "/repo/src/app.ts" }, true)).toBe("Reading app.ts");
    expect(line("exec_shell_command", { command: "npm test" })).toBe("Ran npm test");
    expect(line("grep_search", { pattern: "TODO" })).toBe("Searched for 'TODO'");
    expect(line("file_glob_search", { include: "**" })).toBe("Listed files");
  });

  it("keep the command in the line while a later argument still streams", () => {
    expect(line("exec_shell_command", '{"command": "ls -la", "timeout": 1')).toBe("Ran ls -la");
  });
});

describe("createTracePhraser", () => {
  it("shows the model's own description of a call as it wrote it", () => {
    expect(line("exec_shell_command", { command: "npm test", description: "Run the tests." })).toBe("Run the tests");
  });

  it("reads any other tool as Used, with its main argument", () => {
    expect(line("create_issue", { title: "Bug", query: "crash" })).toBe("Used create_issue crash");
  });

  it("asks the app's resolvers first, then its phrasing table", () => {
    const app = createTracePhraser({
      phrasings: {
        get_datetime: { object: "the time", past: "Checked", present: "Checking", summary: ToolCallSummaryKind.OTHER },
      },
      resolvers: [
        (c) =>
          c.name === "read_file"
            ? { phrasing: { object: "notes", past: "Opened", present: "Opening", summary: ToolCallSummaryKind.READ } }
            : undefined,
      ],
    });

    expect(formatToolCallLine(app.describeToolCall("read_file", '{"path":"a"}'))).toBe("Opened notes");
    expect(formatToolCallLine(app.describeToolCall("get_datetime", "{}"))).toBe("Checked the time");
  });

  it("summarizes a run by kind, other tools last", () => {
    const steps = [
      call("create_issue", {}),
      call("exec_shell_command", { command: "a" }),
      call("read_file", { path: "x" }),
      call("exec_shell_command", { command: "b" }),
      { type: TraceStepType.TEXT },
    ];

    expect(phraser.summarizeToolCalls(steps)).toBe("Ran 2 commands, read 1 file, used 1 tool");
  });

  it("describes what is running now, or the model thinking", () => {
    const steps = [call("read_file", { path: "a.ts" }), call("grep_search", { pattern: "x" }, TraceStepType.TOOL_CALL_PENDING)];

    expect(phraser.describeToolActivity(steps, true)).toBe("Searching for 'x'");
    expect(phraser.describeToolActivity(steps, false)).toBeNull();
    expect(phraser.describeToolActivity([{ type: TraceStepType.REASONING_PENDING }], true)).toBe("Thinking");
  });
});

describe("groupToolCallRuns", () => {
  it("groups calls with the reasoning around them, up to the next text", () => {
    const steps: TraceStep[] = [
      { type: TraceStepType.REASONING },
      call("read_file", {}),
      { type: TraceStepType.REASONING },
      call("grep_search", {}),
      { type: TraceStepType.TEXT },
      call("read_file", {}),
    ];

    expect(groupToolCallRuns(steps, [0, 1, 2, 3, 4, 5], 2)).toEqual([[0, 1, 2, 3], 4, 5]);
  });
});

describe("tool arguments", () => {
  it("read as fields only when every value is short and on one line", () => {
    expect(simpleToolArgFields('{"max_results": 5, "query": "llama"}')).toEqual([
      { label: "Max results", value: "5" },
      { label: "Query", value: "llama" },
    ]);
    expect(simpleToolArgFields('{"body": "a\\nb"}')).toBeNull();
    expect(humanizeArgKey("filePath")).toBe("File path");
  });
});
