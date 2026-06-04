import { apiFetch } from '$lib/utils';
import { API_TOOLS } from '$lib/constants';
import { ToolResponseField } from '$lib/enums';
import type {
	BuiltinToolExecutionContext,
	ToolExecutionResult,
	ServerBuiltinToolInfo
} from '$lib/types';

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
	 */
	static async executeTool(
		toolName: string,
		params: Record<string, unknown>,
		context?: BuiltinToolExecutionContext,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		const result = await apiFetch<Record<string, unknown>>(API_TOOLS.EXECUTE, {
			method: 'POST',
			body: JSON.stringify({ tool: toolName, params, context }),
			signal
		});

		if (ToolResponseField.ERROR in result) {
			return { content: String(result[ToolResponseField.ERROR]), isError: true };
		}

		if (result.status === 'awaiting_user' && typeof result.request_id === 'string') {
			return {
				content: '',
				isError: false,
				awaitingUser: {
					kind: typeof result.kind === 'string' ? result.kind : 'unknown',
					requestID: result.request_id,
					payload:
						typeof result.payload === 'object' && result.payload !== null
							? (result.payload as Record<string, unknown>)
							: {}
				}
			};
		}

		if (ToolResponseField.PLAIN_TEXT in result) {
			return {
				content: String(result[ToolResponseField.PLAIN_TEXT]),
				isError: result.is_error === true
			};
		}

		return { content: JSON.stringify(result), isError: result.is_error === true };
	}
}
