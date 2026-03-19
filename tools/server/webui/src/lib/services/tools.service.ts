import { apiFetch } from '$lib/utils';
import { API_TOOLS } from '$lib/constants';
import type { OpenAIToolDefinition } from '$lib/types';

export class ToolsService {
	/**
	 * Fetch the list of built-in tools from the server.
	 *
	 * @returns Array of tool definitions in OpenAI-compatible format
	 */
	static async list(): Promise<OpenAIToolDefinition[]> {
		return apiFetch<OpenAIToolDefinition[]>(API_TOOLS.LIST);
	}
}
