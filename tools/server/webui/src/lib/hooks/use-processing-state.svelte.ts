import { slotsService } from '$lib/services';
import { config } from '$lib/stores/settings.svelte';
import type { ApiProcessingState } from '$lib/types/api';

/**
 * useProcessingState - Reactive processing state hook
 *
 * This hook provides reactive access to the processing state of the server.
 * It subscribes to timing data updates from the slots service and provides
 * formatted processing details for UI display.
 *
 * @returns {Object} An object containing the processing state, getProcessingMessage,
 * getProcessingDetails, shouldShowDetails, startMonitoring, and stopMonitoring.
 */
export function useProcessingState() {
	let isMonitoring = $state(false);
	let processingState = $state<ApiProcessingState | null>(null);
	let unsubscribe: (() => void) | null = null;

	async function startMonitoring(): Promise<void> {
		if (isMonitoring) return;

		isMonitoring = true;

		unsubscribe = slotsService.subscribe((state) => {
			processingState = state;
		});

		try {
			const currentState = await slotsService.getCurrentState();

			if (currentState) {
				processingState = currentState;
			}

			if (slotsService.isStreaming()) {
				slotsService.startStreaming();
			}
		} catch (error) {
			console.warn('Failed to start slots monitoring:', error);
			// Continue without slots monitoring - graceful degradation
		}
	}

	function stopMonitoring(): void {
		if (!isMonitoring) return;

		isMonitoring = false;
		processingState = null;

		if (unsubscribe) {
			unsubscribe();
			unsubscribe = null;
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
				return 'Preparing response...';
			case 'generating':
				if (processingState.tokensDecoded > 0) {
					return `Generating... (${processingState.tokensDecoded} tokens)`;
				}
				return 'Generating...';
			default:
				return 'Processing...';
		}
	}

	function getProcessingDetails(): string[] {
		if (!processingState) {
			return [];
		}

		const details: string[] = [];
		const currentConfig = config(); // Get fresh config each time

		// Always show context info when we have valid data
		if (processingState.contextUsed >= 0 && processingState.contextTotal > 0) {
			const contextPercent = Math.round(
				(processingState.contextUsed / processingState.contextTotal) * 100
			);

			details.push(
				`Context: ${processingState.contextUsed}/${processingState.contextTotal} (${contextPercent}%)`
			);
		}

		if (processingState.outputTokensUsed > 0) {
			const outputPercent = Math.round(
				(processingState.outputTokensUsed / processingState.outputTokensMax) * 100
			);

			details.push(
				`Output: ${processingState.outputTokensUsed}/${processingState.outputTokensMax} (${outputPercent}%)`
			);
		}

		if (
			currentConfig.showTokensPerSecond &&
			processingState.tokensPerSecond &&
			processingState.tokensPerSecond > 0
		) {
			details.push(`${processingState.tokensPerSecond.toFixed(1)} tokens/sec`);
		}

		if (processingState.speculative) {
			details.push('Speculative decoding enabled');
		}

		return details;
	}

	function shouldShowDetails(): boolean {
		return processingState !== null && processingState.status !== 'idle';
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
