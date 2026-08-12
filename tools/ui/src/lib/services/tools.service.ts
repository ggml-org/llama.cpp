import { base } from '$app/paths';
import { API_TOOLS, X_RESP_TYPE_HEADER, X_TOOL_CWD_HEADER } from '$lib/constants';
import { ToolResponseField } from '$lib/enums';
import type { ServerBuiltinToolInfo, ToolExecutionResult } from '$lib/types';
import { apiFetch } from '$lib/utils';
import { getJsonHeaders } from '$lib/utils/api-headers';
import { parseSseJsonStream, type SseJsonEvent } from '$lib/utils/sse';

export class ToolsService {
	/**
	 * Fetch the list of built-in tools from the server.
	 *
	 * @returns Array of tool definitions in OpenAI-compatible format
	 */
	static async list(): Promise<ServerBuiltinToolInfo[]> {
		return apiFetch<ServerBuiltinToolInfo[]>(API_TOOLS.LIST);
	}

	/**
	 * Execute a built-in tool on the server.
	 *
	 * @param cwd - Working directory for the tool call, sent as the
	 * x-tool-cwd request header. The server resolves relative paths
	 * against it; the model cannot override it.
	 */
	static async executeTool(
		toolName: string,
		params: Record<string, unknown>,
		signal?: AbortSignal,
		cwd?: string
	): Promise<ToolExecutionResult> {
		const result = await apiFetch<Record<string, unknown>>(API_TOOLS.EXECUTE, {
			body: JSON.stringify({ params, tool: toolName }),
			headers: cwd ? { [X_TOOL_CWD_HEADER]: cwd } : undefined,
			method: 'POST',
			signal
		});

		if (ToolResponseField.ERROR in result) {
			return { content: String(result[ToolResponseField.ERROR]), isError: true };
		}

		if (ToolResponseField.PLAIN_TEXT in result) {
			return { content: String(result[ToolResponseField.PLAIN_TEXT]), isError: false };
		}

		return { content: JSON.stringify(result), isError: false };
	}

	/**
	 * Execute a built-in tool and return the raw JSON response. Unlike
	 * executeTool, this preserves structured fields (e.g. file_glob_search's
	 * `entries` and `base`) that the flattened ToolExecutionResult drops.
	 *
	 * @param respType - sent as the x-resp-type request header. Only read_file
	 * honors it, with `base64` to get the raw bytes instead of decoded text.
	 */
	static async executeToolRaw(
		toolName: string,
		params: Record<string, unknown>,
		signal?: AbortSignal,
		cwd?: string,
		respType?: string
	): Promise<Record<string, unknown>> {
		const headers: Record<string, string> = {};

		if (cwd) headers[X_TOOL_CWD_HEADER] = cwd;

		if (respType) headers[X_RESP_TYPE_HEADER] = respType;

		return apiFetch<Record<string, unknown>>(API_TOOLS.EXECUTE, {
			body: JSON.stringify({ params, tool: toolName }),
			headers: Object.keys(headers).length > 0 ? headers : undefined,
			method: 'POST',
			signal
		});
	}

	/**
	 * Stream a built-in tool's output chunks from the server. The server
	 * `POST /tools` endpoint with `{stream: true}` emits `data: {"chunk": "..."}`
	 * events followed by a terminal `data: {"done": true}` (optionally with
	 * `error`). Yields the chunk string for each partial event.
	 *
	 * The terminal event's `error` field, if present, is yielded as a final
	 * synthetic chunk prefixed with an error marker so the accumulated content
	 * already carries the failure context for the caller.
	 *
	 * Throws synchronously if the server rejects the request (e.g. tool does
	 * not support streaming, or 4xx/5xx response). The HTTP fetch goes through
	 * a minimal text/event-stream reader since the chat SSE parser in
	 * chat.service.ts embeds extra resume logic that is unnecessary here.
	 */
	static async *streamTool(
		toolName: string,
		params: Record<string, unknown>,
		signal?: AbortSignal,
		cwd?: string
	): AsyncGenerator<ToolStreamEvent> {
		const headers = getJsonHeaders();

		if (cwd) headers[X_TOOL_CWD_HEADER] = cwd;

		const response = await fetch(`${base}${API_TOOLS.EXECUTE}`, {
			body: JSON.stringify({ params, stream: true, tool: toolName }),
			headers,
			method: 'POST',
			signal
		});

		if (!response.ok || !response.body) {
			const detail = await formatNonOkResponse(response);

			throw new Error(detail);
		}

		const iterator = parseSseJsonStream<ToolServerEvent>(response, signal);

		while (true) {
			const next: IteratorResult<SseJsonEvent<ToolServerEvent>> = await iterator.next();

			if (next.done) return;

			const event = next.value.data;

			if (event.chunk !== undefined) {
				yield { chunk: event.chunk, done: false };
			}

			if (event.done) {
				yield { chunk: null, done: true, error: event.error };

				return;
			}
		}
	}
}

/**
 * One event from streaming a tool's output.
 *   - During execution: `chunk` is a non-empty text fragment, `done: false`.
 *   - On terminal event: `done: true`, `error` populated if the call failed,
 *     and `chunk` is null.
 */
export interface ToolStreamEvent {
	chunk: string | null;
	done: boolean;
	error?: string;
}

/** Wire shape of one SSE event from `POST /tools?stream=true`. */
interface ToolServerEvent {
	chunk?: string;
	done?: boolean;
	error?: string;
}

async function formatNonOkResponse(response: Response): Promise<string> {
	const status = `${response.status} ${response.statusText}`.trim();

	try {
		const errBody = (await response.clone().json()) as { error?: string; message?: string };

		if (errBody?.error) return `${status}: ${errBody.error}`;

		if (errBody?.message) return `${status}: ${errBody.message}`;
	} catch (error) {
		console.error('[tools] Non-JSON error response, falling back to raw text:', error);
		try {
			const text = await response.text();

			if (text.trim()) return `${status}: ${text.trim()}`;
		} catch (error) {
			console.error('[tools] Failed to read error response as text:', error);
		}
	}

	return status || `HTTP ${response.status}`;
}
