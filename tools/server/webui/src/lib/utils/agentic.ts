import type { ApiChatMessageData } from '$lib/types/api';
import type { AgenticMessage, AgenticConfig } from '$lib/types/agentic';
import type { SettingsConfigType } from '$lib/types/settings';
import type { McpServerOverride } from '$lib/types/database';
import { DEFAULT_AGENTIC_CONFIG } from '$lib/constants/agentic';
import { normalizePositiveNumber } from '$lib/utils/number';
import { mcpStore } from '$lib/stores/mcp.svelte';

/**
 * Gets the current agentic configuration.
 * Automatically disables agentic mode if no MCP servers are configured.
 * @param settings - Global settings configuration
 * @param perChatOverrides - Optional per-chat MCP server overrides
 */
export function getAgenticConfig(
	settings: SettingsConfigType,
	perChatOverrides?: McpServerOverride[]
): AgenticConfig {
	const maxTurns = normalizePositiveNumber(
		settings.agenticMaxTurns,
		DEFAULT_AGENTIC_CONFIG.maxTurns
	);
	const maxToolPreviewLines = normalizePositiveNumber(
		settings.agenticMaxToolPreviewLines,
		DEFAULT_AGENTIC_CONFIG.maxToolPreviewLines
	);
	const filterReasoningAfterFirstTurn =
		typeof settings.agenticFilterReasoningAfterFirstTurn === 'boolean'
			? settings.agenticFilterReasoningAfterFirstTurn
			: DEFAULT_AGENTIC_CONFIG.filterReasoningAfterFirstTurn;

	return {
		enabled: mcpStore.hasEnabledServers(perChatOverrides) && DEFAULT_AGENTIC_CONFIG.enabled,
		maxTurns,
		maxToolPreviewLines,
		filterReasoningAfterFirstTurn
	};
}

/**
 * Converts API messages to agentic format.
 */
export function toAgenticMessages(messages: ApiChatMessageData[]): AgenticMessage[] {
	return messages.map((message) => {
		if (message.role === 'assistant' && message.tool_calls && message.tool_calls.length > 0) {
			return {
				role: 'assistant',
				content: message.content,
				tool_calls: message.tool_calls.map((call, index) => ({
					id: call.id ?? `call_${index}`,
					type: (call.type as 'function') ?? 'function',
					function: {
						name: call.function?.name ?? '',
						arguments: call.function?.arguments ?? ''
					}
				}))
			} satisfies AgenticMessage;
		}

		if (message.role === 'tool' && message.tool_call_id) {
			return {
				role: 'tool',
				tool_call_id: message.tool_call_id,
				content: typeof message.content === 'string' ? message.content : ''
			} satisfies AgenticMessage;
		}

		return {
			role: message.role,
			content: message.content
		} satisfies AgenticMessage;
	});
}
