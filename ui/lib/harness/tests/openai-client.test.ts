import { describe, expect, it } from "vitest";
import { createOpenAIClient, type InferenceEvent } from "../src/index.js";

function sse(...chunks: unknown[]): string {
  return chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";
}

// answers every request with the given body, cut into pieces to cross read boundaries
function fakeFetch(body: string, init: ResponseInit = {}, pieceLength = 7) {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const fetch = (async (url: string, requestInit: RequestInit) => {
    calls.push({ init: requestInit, url });

    const bytes = new TextEncoder().encode(body);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (let i = 0; i < bytes.length; i += pieceLength) {
          controller.enqueue(bytes.slice(i, i + pieceLength));
        }

        controller.close();
      },
    });

    return new Response(stream, init);
  }) as typeof globalThis.fetch;

  return { calls, fetch };
}

async function collect(stream: AsyncIterable<InferenceEvent>): Promise<InferenceEvent[]> {
  const events: InferenceEvent[] = [];

  for await (const event of stream) events.push(event);

  return events;
}

const delta = (fields: object, rest: object = {}) => ({
  choices: [{ delta: fields }],
  id: "cmpl-1",
  model: "m",
  ...rest,
});

describe("createOpenAIClient", () => {
  it("posts a streaming request to the chat completions endpoint", async () => {
    const { calls, fetch } = fakeFetch(sse());
    const client = createOpenAIClient({
      apiKey: "key",
      baseUrl: "http://localhost:8080/",
      fetch,
      model: "m",
      params: { temperature: 0 },
    });

    await collect(client.stream({ messages: [{ content: "hi", role: "user" }] }));

    expect(calls[0].url).toBe("http://localhost:8080/v1/chat/completions");
    expect(calls[0].init.headers).toMatchObject({ Authorization: "Bearer key" });
    expect(JSON.parse(calls[0].init.body as string)).toEqual({
      messages: [{ content: "hi", role: "user" }],
      model: "m",
      stream: true,
      temperature: 0,
    });
  });

  it("turns the chunks into events", async () => {
    const timings = { predicted_n: 2 };
    const { fetch } = fakeFetch(
      sse(
        delta({ reasoning_content: "hm" }),
        delta({ content: "Hel" }),
        delta({ content: "lo" }),
        delta({}, { timings }),
      ),
    );
    const events = await collect(
      createOpenAIClient({ baseUrl: "http://x", fetch }).stream({ messages: [] }),
    );

    expect(events).toEqual([
      { id: "cmpl-1", type: "completion_id" },
      { model: "m", type: "model" },
      { delta: "hm", type: "reasoning" },
      { delta: "Hel", type: "text" },
      { delta: "lo", type: "text" },
      { progress: undefined, timings, type: "timings" },
    ]);
  });

  it("merges tool call deltas by index", async () => {
    const { fetch } = fakeFetch(
      sse(
        delta({
          tool_calls: [
            { function: { arguments: "", name: "read" }, id: "a", index: 0, type: "function" },
          ],
        }),
        delta({ tool_calls: [{ function: { arguments: '{"path"' }, index: 0 }] }),
        delta({
          tool_calls: [
            { function: { arguments: ':"x"}' }, index: 0 },
            { function: { arguments: "{}", name: "list" }, id: "b", index: 1, type: "function" },
          ],
        }),
      ),
    );
    const events = await collect(
      createOpenAIClient({ baseUrl: "http://x", fetch }).stream({ messages: [] }),
    );
    const toolCalls = events.filter((event) => event.type === "tool_calls");

    expect(toolCalls).toHaveLength(3);
    expect(toolCalls[2].toolCalls).toEqual([
      { function: { arguments: '{"path":"x"}', name: "read" }, id: "a", type: "function" },
      { function: { arguments: "{}", name: "list" }, id: "b", type: "function" },
    ]);
  });

  it("throws the server's error message", async () => {
    const { fetch } = fakeFetch(JSON.stringify({ error: { message: "no model" } }), {
      status: 400,
    });
    const stream = createOpenAIClient({ baseUrl: "http://x", fetch }).stream({ messages: [] });

    await expect(collect(stream)).rejects.toThrow("HTTP 400: no model");
  });
});
