import { apiFetch } from '$lib/utils';
import { API_TOOLS } from '$lib/constants';
import type { OpenAIToolDefinition, ToolExecutionResult } from '$lib/types';

export class ToolsService {
	/**
	 * Fetch the list of built-in tools from the server.
	 *
	 * @returns Array of tool definitions in OpenAI-compatible format
	 */
	static async list(): Promise<OpenAIToolDefinition[]> {
		return apiFetch<OpenAIToolDefinition[]>(API_TOOLS.LIST);
	}

	/**
	 * Execute a built-in tool on the server.
	 */
	static async executeTool(
		toolName: string,
		params: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		const result = await apiFetch<Record<string, unknown>>(API_TOOLS.EXECUTE, {
			method: 'POST',
			body: JSON.stringify({ tool: toolName, params }),
			signal
		});
		const isError = 'error' in result;
		const content = isError ? String(result.error) : JSON.stringify(result);
		return { content, isError };
	}
}
