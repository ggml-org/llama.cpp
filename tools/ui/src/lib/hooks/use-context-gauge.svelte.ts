/**
 * Reactive state for the context usage gauge: resolves the active model,
 * fetches its cached props, parses live server stats, and exposes per-turn
 * read / fresh / cache / output and cumulative token counts.
 */

import { useProcessingState } from './use-processing-state.svelte';
import {
	type ColorLevel,
	colorLevelFromPercent
} from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/context-gauge';
import { STATS_UNITS } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import { chatStore } from '$lib/stores/chat.svelte';
import { activeMessages } from '$lib/stores/conversations.svelte';
import {
	modelOptions,
	modelsStore,
	selectedModelId,
	singleModelName
} from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';
import type { ChatMessageTimings, DatabaseMessage } from '$lib/types';

interface LiveStats {
	freshTokens: number;
	promptTokens: number;
	cacheTokens: number;
	outputTokens: number;
}

export interface UseContextGaugeReturn {
	readonly activeModelId: string | null;
	readonly isActiveModelLoaded: boolean;
	readonly isActiveModelLoading: boolean;
	readonly contextTotal: number | null;
	readonly contextUsed: number;
	readonly currentRead: number;
	readonly currentFresh: number;
	readonly currentCache: number;
	readonly currentOutput: number;
	readonly kvTotal: number;
	readonly cumulativeRead: number;
	readonly cumulativeOutput: number;
	readonly cumulativeCacheTotal: number;
	readonly averageTokensPerSecond: number | null;
	readonly contextPercent: number | null;
	readonly colorLevel: ColorLevel;
	readonly transientDetails: string[];
	readonly hasAnyUsage: boolean;
	loadModel(): Promise<void>;
	startMonitoring(): void;
}

function lastAssistantTimings(messages: DatabaseMessage[]): ChatMessageTimings | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const m = messages[i];

		if (m.role === MessageRole.ASSISTANT && m.timings) return m.timings;
	}

	return undefined;
}

function deriveLiveStats(
	state: ReturnType<typeof useProcessingState>['processingState']
): LiveStats | null {
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

const TRANSIENT_DETAILS_EXCLUDED_PREFIXES = ['Context:', 'Output:'];

function filterTransientDetails(raw: string[]): string[] {
	return raw.filter((detail) => {
		if (TRANSIENT_DETAILS_EXCLUDED_PREFIXES.some((prefix) => detail.startsWith(prefix))) {
			return false;
		}

		return !detail.includes(STATS_UNITS.TOKENS_PER_SECOND);
	});
}

export function useContextGauge(): UseContextGaugeReturn {
	const processingState = useProcessingState();
	// Resolve the model the gauge reports context for: explicit selection >
	// last assistant model > single-model mode (mirrors useChatScreenActiveModel).
	const activeModelId = $derived.by(() => {
		if (!isRouterMode()) {
			return singleModelName();
		}

		const selectedId = selectedModelId();

		if (selectedId) {
			const model = modelOptions().find((m) => m.id === selectedId);

			if (model) return model.model;
		}

		return chatStore.getConversationModel(activeMessages() as DatabaseMessage[]);
	});
	const isActiveModelLoaded = $derived(
		activeModelId !== null && (!isRouterMode() || modelsStore.isModelLoaded(activeModelId))
	);
	const isActiveModelLoading = $derived(
		activeModelId !== null && modelsStore.isModelOperationInProgress(activeModelId)
	);

	// Pull /props on demand so n_ctx surfaces before the first chat request.
	$effect(() => {
		if (activeModelId && isActiveModelLoaded) {
			const cached = modelsStore.getModelProps(activeModelId);

			if (!cached) {
				void modelsStore.fetchModelProps(activeModelId);
			}
		}
	});

	const contextTotal = $derived.by(() => {
		void modelsStore.propsCacheVersion;

		return activeModelId ? modelsStore.getModelContextSize(activeModelId) : null;
	});
	const liveStats = $derived(deriveLiveStats(processingState.processingState));
	const currentRead = $derived.by(() => {
		const timings = lastAssistantTimings(activeMessages() as DatabaseMessage[]);

		let read = 0;

		if (timings) {
			read = (timings.prompt_n ?? 0) + (timings.cache_n ?? 0);
		}

		// live.promptTokens is already the combined reading (prompt + cache),
		// so do not also add live.cacheTokens.
		if (liveStats && liveStats.promptTokens > 0) {
			read = Math.max(read, liveStats.promptTokens);
		}

		return read;
	});
	const currentFresh = $derived.by(() => {
		const timings = lastAssistantTimings(activeMessages() as DatabaseMessage[]);
		const fresh = timings?.prompt_n ?? 0;

		return Math.max(fresh, liveStats?.freshTokens ?? 0);
	});
	const currentCache = $derived.by(() => {
		const timings = lastAssistantTimings(activeMessages() as DatabaseMessage[]);
		const cached = timings?.cache_n ?? 0;

		if (liveStats && liveStats.promptTokens > 0) {
			return Math.max(cached, liveStats.cacheTokens);
		}

		return cached;
	});
	const currentOutput = $derived.by(() => {
		if (liveStats && liveStats.outputTokens > 0) return liveStats.outputTokens;

		const timings = lastAssistantTimings(activeMessages() as DatabaseMessage[]);

		return timings?.predicted_n ?? 0;
	});
	const kvTotal = $derived(currentRead + currentOutput);
	const contextUsed = $derived(currentRead + currentOutput);
	const cumulative = $derived.by(() => {
		const messages = activeMessages() as DatabaseMessage[];
		// Agentic sessions stamp the same agentic.llm totals onto every
		// assistant message; cache_n is never per-turn so cache_total stays 0.
		const agenticMessages = messages.filter(
			(m) => m.role === MessageRole.ASSISTANT && m.timings?.agentic?.llm?.predicted_n != null
		);

		if (agenticMessages.length > 0) {
			const llm = agenticMessages[agenticMessages.length - 1].timings!.agentic!.llm;
			const output = llm.predicted_n ?? 0;
			const outputMs = llm.predicted_ms ?? 0;
			const averageTokensPerSecond = outputMs > 0 && output > 0 ? (output / outputMs) * 1000 : null;

			return {
				averageTokensPerSecond,
				cacheTotal: 0,
				output,
				read: llm.prompt_n ?? 0
			};
		}

		let read = 0;
		let output = 0;
		let outputMs = 0;
		let cacheTotal = 0;

		for (const m of messages) {
			if (m.role !== MessageRole.ASSISTANT || !m.timings) continue;

			read += m.timings.prompt_n ?? 0;
			cacheTotal += m.timings.cache_n ?? 0;
			output += m.timings.predicted_n ?? 0;
			outputMs += m.timings.predicted_ms ?? 0;
		}
		const averageTokensPerSecond = outputMs > 0 && output > 0 ? (output / outputMs) * 1000 : null;

		return { averageTokensPerSecond, cacheTotal, output, read };
	});
	const contextPercent = $derived.by(() => {
		if (contextTotal === null || contextTotal <= 0) return null;

		return Math.round((contextUsed / contextTotal) * 100);
	});
	const colorLevel = $derived(colorLevelFromPercent(contextPercent));
	// Drop lines the surrounding Context / Output / speed rows already render.
	const transientDetails = $derived(filterTransientDetails(processingState.getTechnicalDetails()));
	const hasAnyUsage = $derived(
		cumulative.read > 0 ||
			cumulative.output > 0 ||
			currentRead > 0 ||
			currentOutput > 0 ||
			cumulative.averageTokensPerSecond !== null ||
			transientDetails.length > 0
	);

	async function loadModel() {
		if (!activeModelId || isActiveModelLoading) return;

		try {
			await modelsStore.loadModel(activeModelId);
		} catch {
			// toast already surfaced by modelsStore.loadModel
		}
	}

	return {
		get activeModelId() {
			return activeModelId;
		},
		get averageTokensPerSecond() {
			return cumulative.averageTokensPerSecond;
		},
		get colorLevel() {
			return colorLevel;
		},
		get contextPercent() {
			return contextPercent;
		},
		get contextTotal() {
			return contextTotal;
		},
		get contextUsed() {
			return contextUsed;
		},
		get cumulativeCacheTotal() {
			return cumulative.cacheTotal;
		},
		get cumulativeOutput() {
			return cumulative.output;
		},
		get cumulativeRead() {
			return cumulative.read;
		},
		get currentCache() {
			return currentCache;
		},
		get currentFresh() {
			return currentFresh;
		},
		get currentOutput() {
			return currentOutput;
		},
		get currentRead() {
			return currentRead;
		},
		get hasAnyUsage() {
			return hasAnyUsage;
		},
		get isActiveModelLoaded() {
			return isActiveModelLoaded;
		},
		get isActiveModelLoading() {
			return isActiveModelLoading;
		},
		get kvTotal() {
			return kvTotal;
		},
		loadModel,
		startMonitoring: () => processingState.startMonitoring(),
		get transientDetails() {
			return transientDetails;
		}
	};
}
