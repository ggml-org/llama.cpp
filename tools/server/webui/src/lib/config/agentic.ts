import { hasEnabledMcpServers } from './mcp';
import type { SettingsConfigType } from '$lib/types/settings';

/**
 * Agentic orchestration configuration.
 */
export interface AgenticConfig {
	enabled: boolean;
	maxTurns: number;
	maxToolPreviewLines: number;
	filterReasoningAfterFirstTurn: boolean;
}

const defaultAgenticConfig: AgenticConfig = {
	enabled: true,
	maxTurns: 100,
	maxToolPreviewLines: 25,
	filterReasoningAfterFirstTurn: true
};

function normalizeNumber(value: unknown, fallback: number): number {
	const parsed = typeof value === 'string' ? Number.parseFloat(value) : Number(value);
	if (!Number.isFinite(parsed) || parsed <= 0) {
		return fallback;
	}

	return parsed;
}

/**
 * Gets the current agentic configuration.
 * Automatically disables agentic mode if no MCP servers are configured.
 */
export function getAgenticConfig(settings: SettingsConfigType): AgenticConfig {
	const maxTurns = normalizeNumber(settings.agenticMaxTurns, defaultAgenticConfig.maxTurns);
	const maxToolPreviewLines = normalizeNumber(
		settings.agenticMaxToolPreviewLines,
		defaultAgenticConfig.maxToolPreviewLines
	);
	const filterReasoningAfterFirstTurn =
		typeof settings.agenticFilterReasoningAfterFirstTurn === 'boolean'
			? settings.agenticFilterReasoningAfterFirstTurn
			: defaultAgenticConfig.filterReasoningAfterFirstTurn;

	return {
		enabled: hasEnabledMcpServers(settings) && defaultAgenticConfig.enabled,
		maxTurns,
		maxToolPreviewLines,
		filterReasoningAfterFirstTurn
	};
}
