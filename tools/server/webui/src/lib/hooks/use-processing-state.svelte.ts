import { activeProcessingState } from '$lib/stores/chat.svelte';
import { config } from '$lib/stores/settings.svelte';

export interface LiveProcessingStats {
	tokensProcessed: number;
	totalTokens: number;
	timeMs: number;
	tokensPerSecond: number;
}

export interface LiveGenerationStats {
	tokensGenerated: number;
	timeMs: number;
	tokensPerSecond: number;
}

export interface UseProcessingStateReturn {
	readonly processingState: ApiProcessingState | null;
	getProcessingDetails(): string[];
	getProcessingMessage(): string;
	getPromptProgressText(): string | null;
	getLiveProcessingStats(): LiveProcessingStats | null;
	getLiveGenerationStats(): LiveGenerationStats | null;
	shouldShowDetails(): boolean;
	startMonitoring(): void;
	stopMonitoring(): void;
}

/**
 * useProcessingState - Reactive processing state hook
 *
 * This hook provides reactive access to the processing state of the server.
 * It directly reads from chatStore's reactive state and provides
 * formatted processing details for UI display.
 *
 * **Features:**
 * - Real-time processing state via direct reactive state binding
 * - Context and output token tracking
 * - Tokens per second calculation
 * - Automatic updates when streaming data arrives
 * - Supports multiple concurrent conversations
 *
 * @returns Hook interface with processing state and control methods
 */
export function useProcessingState(): UseProcessingStateReturn {
	let isMonitoring = $state(false);
	let lastKnownState = $state<ApiProcessingState | null>(null);
	let lastKnownProcessingStats = $state<LiveProcessingStats | null>(null);

	let baseEtaSeconds = $state<number | null>(null); // ETA calculated from last progress update
	let etaCalculatedAt = $state<number | null>(null); // Timestamp when ETA was calculated
	let etaTick = $state(0); // Counter to force re-renders for smooth countdown

	// Derive processing state reactively from chatStore's direct state
	const processingState = $derived.by(() => {
		if (!isMonitoring) {
			return lastKnownState;
		}
		// Read directly from the reactive state export
		return activeProcessingState();
	});

	// Track last known state for keepStatsVisible functionality
	$effect(() => {
		if (processingState && isMonitoring) {
			lastKnownState = processingState;
		}
	});

	// Track last known processing stats for when promptProgress disappears
	$effect(() => {
		if (processingState?.promptProgress) {
			const { processed, total, time_ms, cache } = processingState.promptProgress;
			const actualProcessed = processed - cache;
			const actualTotal = total - cache;

			if (actualProcessed > 0 && time_ms > 0) {
				const tokensPerSecond = actualProcessed / (time_ms / 1000);
				lastKnownProcessingStats = {
					tokensProcessed: actualProcessed,
					totalTokens: actualTotal,
					timeMs: time_ms,
					tokensPerSecond
				};
			}
		}
	});

	// Calculate base ETA when progress data arrives
	// This only sets the baseline - actual display uses time-based calculation
	$effect(() => {
		if (processingState?.promptProgress) {
			const { processed, total, time_ms, cache } = processingState.promptProgress;
			const actualProcessed = processed - cache;
			const actualTotal = total - cache;

			if (actualProcessed > 0 && time_ms > 0) {
				const tokensPerSec = actualProcessed / (time_ms / 1000);
				const remaining = actualTotal - actualProcessed;

				baseEtaSeconds = Math.ceil(remaining / tokensPerSec);
				etaCalculatedAt = Date.now();
			}
		} else {
			baseEtaSeconds = null;
			etaCalculatedAt = null;
		}
	});

	// Timer to force re-renders for smooth countdown display
	// Does NOT modify ETA value - only triggers reactive updates
	$effect(() => {
		if (baseEtaSeconds === null) return;

		const interval = setInterval(() => {
			etaTick++; // Force re-render
		}, 1000);

		return () => clearInterval(interval);
	});

	function getEtaSecondsRemaining(): number | null {
		void etaTick;

		if (baseEtaSeconds === null || etaCalculatedAt === null) return null;

		const elapsedSeconds = Math.floor((Date.now() - etaCalculatedAt) / 1000);
		const remaining = baseEtaSeconds - elapsedSeconds;

		return Math.max(0, remaining);
	}

	function startMonitoring(): void {
		if (isMonitoring) return;
		isMonitoring = true;
	}

	function stopMonitoring(): void {
		if (!isMonitoring) return;
		isMonitoring = false;

		// Only clear last known state if keepStatsVisible is disabled
		const currentConfig = config();
		if (!currentConfig.keepStatsVisible) {
			lastKnownState = null;
			lastKnownProcessingStats = null;
		}
	}

	function getProcessingMessage(): string {
		if (!processingState) {
			return 'Processing...';
		}

		switch (processingState.status) {
			case 'initializing':
				return 'Initializing...';
			case 'preparing':
				if (processingState.progressPercent !== undefined) {
					return `Processing (${processingState.progressPercent}%)`;
				}
				return 'Preparing response...';
			case 'generating':
				return '';
			default:
				return 'Processing...';
		}
	}

	function getProcessingDetails(): string[] {
		// Use current processing state or fall back to last known state
		const stateToUse = processingState || lastKnownState;
		if (!stateToUse) {
			return [];
		}

		const details: string[] = [];

		// Always show context info when we have valid data
		if (stateToUse.contextUsed >= 0 && stateToUse.contextTotal > 0) {
			const contextPercent = Math.round((stateToUse.contextUsed / stateToUse.contextTotal) * 100);

			details.push(
				`Context: ${stateToUse.contextUsed}/${stateToUse.contextTotal} (${contextPercent}%)`
			);
		}

		if (stateToUse.outputTokensUsed > 0) {
			// Handle infinite max_tokens (-1) case
			if (stateToUse.outputTokensMax <= 0) {
				details.push(`Output: ${stateToUse.outputTokensUsed}/∞`);
			} else {
				const outputPercent = Math.round(
					(stateToUse.outputTokensUsed / stateToUse.outputTokensMax) * 100
				);

				details.push(
					`Output: ${stateToUse.outputTokensUsed}/${stateToUse.outputTokensMax} (${outputPercent}%)`
				);
			}
		}

		if (stateToUse.tokensPerSecond && stateToUse.tokensPerSecond > 0) {
			details.push(`${stateToUse.tokensPerSecond.toFixed(1)} tokens/sec`);
		}

		if (stateToUse.speculative) {
			details.push('Speculative decoding enabled');
		}

		return details;
	}

	function shouldShowDetails(): boolean {
		return processingState !== null && processingState.status !== 'idle';
	}

	/**
	 * Returns a short progress message percent and ETA
	 */
	function getPromptProgressText(): string | null {
		if (!processingState?.promptProgress) return null;

		const { processed, total, cache } = processingState.promptProgress;

		const actualProcessed = processed - cache;
		const actualTotal = total - cache;
		const percent = Math.round((actualProcessed / actualTotal) * 100);

		const etaSeconds = getEtaSecondsRemaining();
		if (etaSeconds === null || etaSeconds <= 0) {
			return `Processing ${percent}%`;
		}

		const etaDisplay =
			etaSeconds >= 60 ? `${Math.floor(etaSeconds / 60)}m${etaSeconds % 60}s` : `${etaSeconds}s`;

		return `Processing ${percent}% (${etaDisplay} left)`;
	}

	/**
	 * Returns live processing statistics for display (prompt processing phase)
	 * Returns last known stats when promptProgress becomes unavailable
	 */
	function getLiveProcessingStats(): LiveProcessingStats | null {
		if (processingState?.promptProgress) {
			const { processed, total, time_ms, cache } = processingState.promptProgress;

			const actualProcessed = processed - cache;
			const actualTotal = total - cache;

			if (actualProcessed > 0 && time_ms > 0) {
				const tokensPerSecond = actualProcessed / (time_ms / 1000);

				return {
					tokensProcessed: actualProcessed,
					totalTokens: actualTotal,
					timeMs: time_ms,
					tokensPerSecond
				};
			}
		}

		// Return last known stats if promptProgress is no longer available
		return lastKnownProcessingStats;
	}

	/**
	 * Returns live generation statistics for display (token generation phase)
	 */
	function getLiveGenerationStats(): LiveGenerationStats | null {
		if (!processingState) return null;

		const { tokensDecoded, tokensPerSecond } = processingState;

		if (tokensDecoded <= 0) return null;

		// Calculate time from tokens and speed
		const timeMs =
			tokensPerSecond && tokensPerSecond > 0 ? (tokensDecoded / tokensPerSecond) * 1000 : 0;

		return {
			tokensGenerated: tokensDecoded,
			timeMs,
			tokensPerSecond: tokensPerSecond || 0
		};
	}

	return {
		get processingState() {
			return processingState;
		},
		getProcessingDetails,
		getProcessingMessage,
		getPromptProgressText,
		getLiveProcessingStats,
		getLiveGenerationStats,
		shouldShowDetails,
		startMonitoring,
		stopMonitoring
	};
}
