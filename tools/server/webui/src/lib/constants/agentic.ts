import type { AgenticConfig } from '$lib/types/agentic';

export const DEFAULT_AGENTIC_CONFIG: AgenticConfig = {
	enabled: true,
	maxTurns: 100,
	maxToolPreviewLines: 25,
	filterReasoningAfterFirstTurn: true
} as const;
