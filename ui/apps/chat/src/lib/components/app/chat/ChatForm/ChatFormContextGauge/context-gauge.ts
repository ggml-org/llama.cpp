import { ColorLevel, MessageRole } from '$lib/enums';
import type { ApiProcessingState, DatabaseMessage } from '$lib/types';

const WARNING_THRESHOLD = 80;
const CRITICAL_THRESHOLD = 95;

export function colorLevelFromPercent(percent: number | null): ColorLevel {
	if (percent === null) return ColorLevel.NEUTRAL;

	if (percent >= CRITICAL_THRESHOLD) return ColorLevel.CRITICAL;

	if (percent >= WARNING_THRESHOLD) return ColorLevel.WARNING;

	return ColorLevel.OK;
}

export function colorLevelTextClass(level: ColorLevel): string {
	switch (level) {
		case ColorLevel.CRITICAL:
			return 'text-red-400';
		case ColorLevel.WARNING:
			return 'text-amber-400';
		case ColorLevel.OK:
			return 'text-muted-foreground';
		default:
			return 'text-muted-foreground';
	}
}

export function colorLevelBgClass(level: ColorLevel): string {
	switch (level) {
		case ColorLevel.CRITICAL:
			return 'bg-red-500';
		case ColorLevel.WARNING:
			return 'bg-amber-500';
		case ColorLevel.OK:
			return 'bg-green-500';
		default:
			return 'bg-muted';
	}
}

/**
 * Whether the conversation has spent any tokens yet, so the gauge has something to show: the
 * timings its replies recorded, or the live state of a reply still being generated.
 */
export function hasProcessedTokens(
	messages: DatabaseMessage[],
	liveState: ApiProcessingState | null
): boolean {
	let totalHistoricalTokens = 0;

	for (const m of messages) {
		if (m.role !== MessageRole.ASSISTANT) continue;

		const timings = m.timings;

		if (!timings) continue;

		const agenticLlm = timings.agentic?.llm;

		if (agenticLlm?.prompt_n != null || agenticLlm?.predicted_n != null) {
			totalHistoricalTokens += (agenticLlm?.prompt_n ?? 0) + (agenticLlm?.predicted_n ?? 0);
		} else {
			totalHistoricalTokens += (timings.prompt_n ?? 0) + (timings.predicted_n ?? 0);
		}
	}

	if (totalHistoricalTokens > 0) return true;

	if (!liveState) return false;

	const livePromptTokens = Math.max(
		liveState.promptTokens ?? 0,
		liveState.promptProgress?.processed ?? 0
	);
	const liveOutputTokens = liveState.outputTokensUsed ?? 0;

	return livePromptTokens > 0 || liveOutputTokens > 0;
}
