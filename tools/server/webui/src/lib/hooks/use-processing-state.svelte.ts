import { activeProcessingState } from '$lib/stores/chat.svelte';
import { config } from '$lib/stores/settings.svelte';

export interface UseProcessingStateReturn {
	readonly processingState: ApiProcessingState | null;
	getProcessingDetails(): string[];
	getProcessingMessage(): string;
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
		}
	}

	function getProcessingMessage(): string {
		const state = processingState;
		if (!state) {
			return 'Processing...';
		}

		switch (state.status) {
			case 'initializing':
				return 'Initializing...';
			case 'preparing':
				if (state.progressPercent !== undefined) {
					return `Processing (${state.progressPercent}%)`;
				}
				return 'Preparing response...';
			case 'generating':
				if (state.tokensDecoded > 0) {
					return `Generating... (${state.tokensDecoded} tokens)`;
				}
				return 'Generating...';
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
		const state = processingState;
		return state !== null && state.status !== 'idle';
	}

	return {
		get processingState() {
			return processingState;
		},
		getProcessingDetails,
		getProcessingMessage,
		shouldShowDetails,
		startMonitoring,
		stopMonitoring
	};
}
