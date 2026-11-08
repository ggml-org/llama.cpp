/**
 * ContextStatsStore - Context window usage stats for the active conversation
 *
 * Combines token usage persisted in message timings metadata with
 * server-originating data: model context size from /props (modelsStore)
 * and live processing state while streaming (chatStore).
 */

import { MessageRole, MessageType } from '$lib/enums';
// direct imports between stores, not via the barrel, to avoid circular deps
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import type {
	ApiProcessingState,
	ChatMessageAgenticTimings,
	ChatMessageTimings,
	DatabaseMessage
} from '$lib/types';

interface LiveStats {
	freshTokens: number;
	promptTokens: number;
	cacheTokens: number;
	outputTokens: number;
}

interface AssistantTimingsSummary {
	lastAgenticLlm: ChatMessageAgenticTimings['llm'] | undefined;
	lastTimings: ChatMessageTimings | undefined;
	cacheTotal: number;
	output: number;
	outputMs: number;
	read: number;
}

/**
 * One forward pass over the messages computing everything the deriveds
 * below need: the last context bearing timings (per-turn gauges), the last
 * agentic llm totals (cumulative gauge) and the cumulative sums. During
 * streaming activeMessages churns every chunk, and each of these used to be
 * its own O(n) scan re-run per chunk.
 */
function summarizeAssistantTimings(messages: DatabaseMessage[]): AssistantTimingsSummary {
	let lastAgenticLlm: ChatMessageAgenticTimings['llm'] | undefined;
	let lastTimings: ChatMessageTimings | undefined;
	let read = 0;
	let cacheTotal = 0;
	let output = 0;
	let outputMs = 0;

	for (const m of messages) {
		// A stamped compaction node carries the prompt timings of the reduced
		// context, so it supersedes earlier assistant turns as the current
		// context size while staying out of the cumulative sums.
		if (m.type === MessageType.COMPACTION && m.timings) {
			lastTimings = m.timings;

			continue;
		}

		if (m.role !== MessageRole.ASSISTANT || !m.timings) continue;

		lastTimings = m.timings;

		if (m.timings.agentic?.llm?.predicted_n != null) {
			lastAgenticLlm = m.timings.agentic.llm;
		}

		read += m.timings.prompt_n ?? 0;
		cacheTotal += m.timings.cache_n ?? 0;
		output += m.timings.predicted_n ?? 0;
		outputMs += m.timings.predicted_ms ?? 0;
	}

	return { cacheTotal, lastAgenticLlm, lastTimings, output, outputMs, read };
}

function deriveLiveStats(state: ApiProcessingState | null): LiveStats | null {
	if (!state || (state.status !== 'preparing' && state.status !== 'generating')) {
		return null;
	}

	const promptTokens = state.promptTokens ?? 0;
	const cacheTokens = state.cacheTokens ?? 0;

	return {
		cacheTokens,
		freshTokens: promptTokens,
		outputTokens: state.outputTokensUsed ?? 0,
		promptTokens: promptTokens + cacheTokens
	};
}

class ContextStatsStore {
	// The canonical resolution lives in modelsStore.activeModelId.
	activeModelId = $derived(modelsStore.activeModelId);

	// shared by currentRead/Fresh/Cache/Output and cumulative so a per-chunk
	// churn of activeMessages triggers exactly one scan instead of one per
	// derived
	private assistantTimings = $derived.by(() =>
		summarizeAssistantTimings(conversationsStore.activeMessages as DatabaseMessage[])
	);

	private cumulative = $derived.by(() => {
		const convId = conversationsStore.activeConversation?.id;
		// A running agentic flow stamps llm totals on messages only when it
		// exits, so read its live session totals instead.
		const liveLlm = convId ? agenticStore.getLiveLlmTotals(convId) : null;

		if (liveLlm) {
			const outputMs = liveLlm.predicted_ms;
			const averageTokensPerSecond =
				outputMs > 0 && liveLlm.predicted_n > 0 ? (liveLlm.predicted_n / outputMs) * 1000 : null;

			return {
				averageTokensPerSecond,
				cacheTotal: 0,
				output: liveLlm.predicted_n,
				read: liveLlm.prompt_n
			};
		}

		const { cacheTotal, lastAgenticLlm, output, outputMs, read } = this.assistantTimings;

		// Agentic sessions stamp the same agentic.llm totals onto every
		// assistant message; cache_n is never per-turn so cache_total stays 0.
		if (lastAgenticLlm) {
			const averageTokensPerSecond =
				lastAgenticLlm.predicted_ms > 0 && lastAgenticLlm.predicted_n > 0
					? (lastAgenticLlm.predicted_n / lastAgenticLlm.predicted_ms) * 1000
					: null;

			return {
				averageTokensPerSecond,
				cacheTotal: 0,
				output: lastAgenticLlm.predicted_n ?? 0,
				read: lastAgenticLlm.prompt_n ?? 0
			};
		}

		const averageTokensPerSecond = outputMs > 0 && output > 0 ? (output / outputMs) * 1000 : null;

		return { averageTokensPerSecond, cacheTotal, output, read };
	});

	averageTokensPerSecond = $derived(this.cumulative.averageTokensPerSecond);

	contextTotal = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return this.activeModelId ? modelsStore.props.getModelContextSize(this.activeModelId) : null;
	});

	private liveStats = $derived(deriveLiveStats(chatStore.processing.activeState));

	currentOutput = $derived.by(() => {
		if (this.liveStats && this.liveStats.outputTokens > 0) return this.liveStats.outputTokens;

		return this.assistantTimings.lastTimings?.predicted_n ?? 0;
	});

	currentRead = $derived.by(() => {
		const timings = this.assistantTimings.lastTimings;

		let read = 0;

		if (timings) {
			read = (timings.prompt_n ?? 0) + (timings.cache_n ?? 0);
		}

		// live.promptTokens is already the combined reading (prompt + cache),
		// so do not also add live.cacheTokens.
		if (this.liveStats && this.liveStats.promptTokens > 0) {
			read = Math.max(read, this.liveStats.promptTokens);
		}

		return read;
	});

	contextUsed = $derived(this.currentRead + this.currentOutput);

	contextAvailable = $derived(
		this.contextTotal !== null ? this.contextTotal - this.contextUsed : null
	);

	contextPercent = $derived.by(() => {
		if (this.contextTotal === null || this.contextTotal <= 0) return null;

		return Math.round((this.contextUsed / this.contextTotal) * 100);
	});

	cumulativeCacheTotal = $derived(this.cumulative.cacheTotal);

	cumulativeOutput = $derived(this.cumulative.output);

	cumulativeRead = $derived(this.cumulative.read);

	currentCache = $derived.by(() => {
		const cached = this.assistantTimings.lastTimings?.cache_n ?? 0;

		if (this.liveStats && this.liveStats.promptTokens > 0) {
			return Math.max(cached, this.liveStats.cacheTokens);
		}

		return cached;
	});

	currentFresh = $derived.by(() => {
		const fresh = this.assistantTimings.lastTimings?.prompt_n ?? 0;

		return Math.max(fresh, this.liveStats?.freshTokens ?? 0);
	});

	isActiveModelLoaded = $derived(
		this.activeModelId !== null &&
			(!serverStore.isRouterMode || modelsStore.isModelLoaded(this.activeModelId))
	);

	isActiveModelLoading = $derived(
		this.activeModelId !== null && modelsStore.status.isOperationInProgress(this.activeModelId)
	);

	kvTotal = $derived(this.currentRead + this.currentOutput);
}

export const contextStatsStore = new ContextStatsStore();
