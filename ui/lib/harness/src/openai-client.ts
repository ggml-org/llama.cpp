import type { InferenceClient, InferenceEvent, PartialToolCall } from "./types.js";

const CHAT_COMPLETIONS_PATH = "/v1/chat/completions";
const SSE_DATA_PREFIX = "data:";
const SSE_DONE = "[DONE]";

export interface OpenAIClientOptions {
  /** Server root, e.g. `http://localhost:8080`. */
  baseUrl: string;
  apiKey?: string;
  model?: string;
  /** Extra fields of the request body, e.g. `temperature`. */
  params?: Record<string, unknown>;
  headers?: Record<string, string>;
  /** Default: the global fetch. */
  fetch?: typeof fetch;
}

interface ToolCallDelta extends PartialToolCall {
  index?: number;
}

function mergeToolCallDeltas(toolCalls: PartialToolCall[], deltas: ToolCallDelta[]): void {
  for (const delta of deltas) {
    // without an index, a delta with an id opens a call and any other one continues the last call
    const index =
      delta.index ?? (delta.id || toolCalls.length === 0 ? toolCalls.length : toolCalls.length - 1);
    const call = (toolCalls[index] ??= { function: { arguments: "", name: "" } });

    if (delta.id) call.id = delta.id;

    if (delta.type) call.type = delta.type;

    call.function = {
      arguments: (call.function?.arguments ?? "") + (delta.function?.arguments ?? ""),
      name: (call.function?.name ?? "") + (delta.function?.name ?? ""),
    };
  }
}

async function errorFromResponse(response: Response): Promise<Error> {
  const body = await response.text().catch(() => "");

  let message = body;

  try {
    message = JSON.parse(body).error?.message ?? body;
  } catch {
    // not JSON: keep the body text
  }

  return new Error(`HTTP ${response.status}${message ? `: ${message}` : ""}`);
}

/**
 * An inference client for llama-server's chat completions endpoint. It reads the `timings` and
 * `prompt_progress` fields llama-server adds.
 */
export function createOpenAIClient(options: OpenAIClientOptions): InferenceClient {
  const { apiKey, baseUrl, headers, model, params } = options;
  const url = baseUrl.replace(/\/+$/, "") + CHAT_COMPLETIONS_PATH;

  return {
    async *stream({ messages, signal, tools }): AsyncGenerator<InferenceEvent> {
      const response = await (options.fetch ?? fetch)(url, {
        body: JSON.stringify({ ...params, messages, model, stream: true, tools }),
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
          ...headers,
        },
        method: "POST",
        signal,
      });

      if (!response.ok) throw await errorFromResponse(response);

      if (!response.body) throw new Error("The response has no body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const toolCalls: PartialToolCall[] = [];

      let buffer = "";
      let completionId: string | undefined;
      let modelName: string | undefined;

      try {
        while (true) {
          const { done, value } = await reader.read();

          if (done) return;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");

          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith(SSE_DATA_PREFIX)) continue;

            const data = line.slice(SSE_DATA_PREFIX.length).trim();

            if (data === SSE_DONE) return;

            const chunk = JSON.parse(data);

            if (chunk.error) throw new Error(chunk.error.message ?? "Stream error");

            if (chunk.id && chunk.id !== completionId) {
              completionId = chunk.id;
              yield { id: chunk.id, type: "completion_id" };
            }

            if (chunk.model && chunk.model !== modelName) {
              modelName = chunk.model;
              yield { model: chunk.model, type: "model" };
            }

            const delta = chunk.choices?.[0]?.delta;
            const reasoning = delta?.reasoning_content ?? delta?.reasoning;

            if (reasoning) yield { delta: reasoning, type: "reasoning" };

            if (delta?.content) yield { delta: delta.content, type: "text" };

            if (delta?.tool_calls?.length) {
              mergeToolCallDeltas(toolCalls, delta.tool_calls);
              yield { toolCalls: toolCalls.map((call) => ({ ...call })), type: "tool_calls" };
            }

            if (chunk.timings || chunk.prompt_progress) {
              yield { progress: chunk.prompt_progress, timings: chunk.timings, type: "timings" };
            }
          }
        }
      } finally {
        reader.cancel().catch(() => {});
      }
    },
  };
}
