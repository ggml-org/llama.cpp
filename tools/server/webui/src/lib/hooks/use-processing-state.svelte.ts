import { slotsService } from '$lib/services';
import { config } from '$lib/stores/settings.svelte';
import type { ApiProcessingState } from '$lib/types/api';

/**
 * useProcessingState - Reactive processing state hook
 *
 * This hook provides reactive access to the processing state of the server.
 * It monitors the slots endpoint for changes and updates the processing state
 * accordingly. The hook also provides functions to start and stop monitoring,
 * as well as a function to get the current processing message and details.
 *
 * @returns {Object} An object containing the processing state, isPolling, getProcessingMessage,
 * getProcessingDetails, shouldShowDetails, startMonitoring, and stopMonitoring.
 */
export function useProcessingState() {
	let isPolling = $state(false);
	let processingState = $state<ApiProcessingState | null>(null);
	let unsubscribe: (() => void) | null = null;

	async function startMonitoring(): Promise<void> {
		if (isPolling) return;

		isPolling = true;

		unsubscribe = slotsService.subscribe((state) => {
			processingState = state;
		});

		try {
			const currentState = await slotsService.getCurrentState();

			if (currentState) {
				processingState = currentState;
			}

			if (slotsService.isStreaming()) {
				slotsService.startStreamingPolling();
			}
		} catch (error) {
			console.warn('Failed to start slots monitoring:', error);
			// Continue without slots monitoring - graceful degradation
		}
	}

	function stopMonitoring(): void {
		if (!isPolling) return;

		isPolling = false;
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

		if (processingState.contextUsed > 0) {
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
		get isPolling() {
			return isPolling;
		},
		getProcessingDetails,
		getProcessingMessage,
		shouldShowDetails,
		startMonitoring,
		stopMonitoring
	};
}
