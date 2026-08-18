/**
 * View layer over contextStatsStore for the context usage gauge: adds
 * color levels, transient detail formatting, on-demand /props fetching
 * and model loading on top of the store's token stats.
 */

import { useProcessingState } from './use-processing-state.svelte';
import { colorLevelFromPercent } from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/context-gauge';
import { STATS_UNITS } from '$lib/constants';
import { ColorLevel } from '$lib/enums';
import { contextStatsStore, modelsStore } from '$lib/stores';

export interface UseContextGaugeReturn {
	readonly activeModelId: string | null;
	readonly isActiveModelLoaded: boolean;
	readonly isActiveModelLoading: boolean;
	readonly contextTotal: number | null;
	readonly contextUsed: number;
	readonly contextAvailable: number | null;
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

	// Pull /props on demand so n_ctx surfaces before the first chat request.
	$effect(() => {
		const modelId = contextStatsStore.activeModelId;

		if (modelId && contextStatsStore.isActiveModelLoaded) {
			const cached = modelsStore.getModelProps(modelId);

			if (!cached) {
				void modelsStore.fetchModelProps(modelId);
			}
		}
	});

	const colorLevel = $derived(colorLevelFromPercent(contextStatsStore.contextPercent));
	// Drop lines the surrounding Context / Output / speed rows already render.
	const transientDetails = $derived(filterTransientDetails(processingState.getTechnicalDetails()));
	const hasAnyUsage = $derived(
		contextStatsStore.cumulativeRead > 0 ||
			contextStatsStore.cumulativeOutput > 0 ||
			contextStatsStore.currentRead > 0 ||
			contextStatsStore.currentOutput > 0 ||
			contextStatsStore.averageTokensPerSecond !== null ||
			transientDetails.length > 0
	);

	async function loadModel() {
		const modelId = contextStatsStore.activeModelId;

		if (!modelId || contextStatsStore.isActiveModelLoading) return;

		try {
			await modelsStore.loadModel(modelId);
		} catch {
			// toast already surfaced by modelsStore.loadModel
		}
	}

	return {
		get activeModelId() {
			return contextStatsStore.activeModelId;
		},
		get averageTokensPerSecond() {
			return contextStatsStore.averageTokensPerSecond;
		},
		get colorLevel() {
			return colorLevel;
		},
		get contextAvailable() {
			return contextStatsStore.contextAvailable;
		},
		get contextPercent() {
			return contextStatsStore.contextPercent;
		},
		get contextTotal() {
			return contextStatsStore.contextTotal;
		},
		get contextUsed() {
			return contextStatsStore.contextUsed;
		},
		get cumulativeCacheTotal() {
			return contextStatsStore.cumulativeCacheTotal;
		},
		get cumulativeOutput() {
			return contextStatsStore.cumulativeOutput;
		},
		get cumulativeRead() {
			return contextStatsStore.cumulativeRead;
		},
		get currentCache() {
			return contextStatsStore.currentCache;
		},
		get currentFresh() {
			return contextStatsStore.currentFresh;
		},
		get currentOutput() {
			return contextStatsStore.currentOutput;
		},
		get currentRead() {
			return contextStatsStore.currentRead;
		},
		get hasAnyUsage() {
			return hasAnyUsage;
		},
		get isActiveModelLoaded() {
			return contextStatsStore.isActiveModelLoaded;
		},
		get isActiveModelLoading() {
			return contextStatsStore.isActiveModelLoading;
		},
		get kvTotal() {
			return contextStatsStore.kvTotal;
		},
		loadModel,
		startMonitoring: () => processingState.startMonitoring(),
		get transientDetails() {
			return transientDetails;
		}
	};
}
