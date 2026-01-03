import { hasEnabledMcpServers } from './mcp';
import type { SettingsConfigType } from '$lib/types/settings';
import type { AgenticConfig } from '$lib/types/agentic';
import type { McpServerOverride } from '$lib/types/database';
import { DEFAULT_AGENTIC_CONFIG } from '$lib/constants/agentic';
import { normalizePositiveNumber } from '$lib/utils/number';

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
		enabled: hasEnabledMcpServers(settings, perChatOverrides) && DEFAULT_AGENTIC_CONFIG.enabled,
		maxTurns,
		maxToolPreviewLines,
		filterReasoningAfterFirstTurn
	};
}
