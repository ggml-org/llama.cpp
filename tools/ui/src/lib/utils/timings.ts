/**
 * Client side timing fallback for backends that do not report their own.
 *
 * llama.cpp streams per-token timings; OpenAI and Anthropic compatible servers
 * do not. Token counts come from the usage block of the final chunk (or the
 * count of streamed deltas as a fallback), times are measured locally: the wait
 * for the first token is attributed to prompt processing, the rest to
 * generation. Wall clock, so network and queueing are part of the numbers.
 */

import type { ApiChatCompletionUsage } from '$lib/types/api';
import type { ChatMessageTimings } from '$lib/types/chat';

export interface StreamClock {
	startedAt: number;
	firstTokenAt: number | null;
	lastTokenAt: number | null;
}

/** Prompt/output token counts, accepting OpenAI and Anthropic usage fields. */
export function usageTokenCounts(usage: ApiChatCompletionUsage | undefined): {
	promptTokens: number;
	completionTokens: number;
} {
	return {
		completionTokens: usage?.completion_tokens ?? usage?.output_tokens ?? 0,
		promptTokens: usage?.prompt_tokens ?? usage?.input_tokens ?? 0
	};
}

export function buildTimingsFromUsage(
	usage: ApiChatCompletionUsage | undefined,
	clock: StreamClock,
	fallbackTokens = 0
): ChatMessageTimings | null {
	const { completionTokens, promptTokens } = usageTokenCounts(usage);
	const predictedN = completionTokens || fallbackTokens;

	if (promptTokens === 0 && predictedN === 0) return null;

	const { firstTokenAt, startedAt } = clock;
	const lastTokenAt = clock.lastTokenAt ?? firstTokenAt;

	return {
		// clamp so a one-token reply still reports a positive duration
		predicted_ms: firstTokenAt && lastTokenAt ? Math.max(1, lastTokenAt - firstTokenAt) : undefined,
		predicted_n: predictedN,
		prompt_ms: firstTokenAt ? Math.max(1, firstTokenAt - startedAt) : undefined,
		prompt_n: promptTokens
	};
}
