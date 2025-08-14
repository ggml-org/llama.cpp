import { slotsService } from '$lib/services/slots';
import type { ApiProcessingState } from '$lib/types/api';

export function useProcessingState() {
	let processingState = $state<ApiProcessingState | null>(null);
	let isPolling = $state(false);
	let unsubscribe: (() => void) | null = null;

	async function startMonitoring(): Promise<void> {
		if (isPolling) return;

		isPolling = true;
		
		unsubscribe = slotsService.subscribe((state) => {
			processingState = state;
		});

		try {
			await slotsService.startPolling();
		} catch (error) {
			console.warn('Failed to start slots polling:', error);
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

		slotsService.stopPolling();
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

		if (processingState.contextUsed > 0) {
			const contextPercent = Math.round((processingState.contextUsed / processingState.contextTotal) * 100);
			details.push(`Context: ${processingState.contextUsed}/${processingState.contextTotal} (${contextPercent}%)`);
		}

		if (processingState.temperature !== 0.8) {
			details.push(`Temperature: ${processingState.temperature.toFixed(1)}`);
		}
		
		if (processingState.topP !== 0.95) {
			details.push(`Top-p: ${processingState.topP.toFixed(2)}`);
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
		get processingState() { return processingState; },
		get isPolling() { return isPolling; },
		startMonitoring,
		stopMonitoring,
		getProcessingMessage,
		getProcessingDetails,
		shouldShowDetails
	};
}
