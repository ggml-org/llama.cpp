/**
 * Client side timing fallback for backends that do not report their own.
 *
 * llama.cpp streams per-token timings; OpenAI-compatible servers
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

/**
 * Prompt/output/cache token counts. `promptTokens` excludes the cache read
 * tokens, which are returned separately as `cacheTokens`, so the two always
 * add up to the prompt size.
 */
export function usageTokenCounts(usage: ApiChatCompletionUsage | undefined): {
	cacheTokens: number;
	completionTokens: number;
	promptTokens: number;
} {
	// a total that includes the cache reads, which are reported separately
	const cacheTokens =
		usage?.prompt_tokens_details?.cached_tokens ??
		usage?.prompt_cache_hit_tokens ??
		usage?.cached_tokens ??
		0;
	const promptTotal = usage?.prompt_tokens ?? 0;

	return {
		cacheTokens,
		completionTokens: usage?.completion_tokens ?? 0,
		promptTokens: Math.max(0, promptTotal - cacheTokens)
	};
}

export function buildTimingsFromUsage(
	usage: ApiChatCompletionUsage | undefined,
	clock: StreamClock,
	fallbackTokens = 0
): ChatMessageTimings | null {
	const { cacheTokens, completionTokens, promptTokens } = usageTokenCounts(usage);
	const predictedN = completionTokens || fallbackTokens;

	if (promptTokens === 0 && predictedN === 0) return null;

	const { firstTokenAt, startedAt } = clock;
	const lastTokenAt = clock.lastTokenAt ?? firstTokenAt;

	return {
		cache_n: cacheTokens,
		// clamp so a one-token reply still reports a positive duration
		predicted_ms: firstTokenAt && lastTokenAt ? Math.max(1, lastTokenAt - firstTokenAt) : undefined,
		predicted_n: predictedN,
		prompt_ms: firstTokenAt ? Math.max(1, firstTokenAt - startedAt) : undefined,
		prompt_n: promptTokens
	};
}
