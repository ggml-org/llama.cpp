import { config } from '$lib/stores/settings.svelte';

/**
 * SlotsService - Real-time processing state monitoring and token rate calculation
 *
 * This service provides real-time information about generation progress, token rates,
 * and context usage based on timing data from ChatService streaming responses.
 * It manages streaming session tracking and provides accurate processing state updates.
 *
 * **Architecture & Relationships:**
 * - **SlotsService** (this class): Processing state monitoring
 *   - Receives timing data from ChatService streaming responses
 *   - Calculates token generation rates and context usage
 *   - Manages streaming session lifecycle
 *   - Provides real-time updates to UI components
 *
 * - **ChatService**: Provides timing data from `/chat/completions` streaming
 * - **UI Components**: Subscribe to processing state for progress indicators
 *
 * **Key Features:**
 * - **Real-time Monitoring**: Live processing state during generation
 * - **Token Rate Calculation**: Accurate tokens/second from timing data
 * - **Context Tracking**: Current context usage and remaining capacity
 * - **Streaming Lifecycle**: Start/stop tracking for streaming sessions
 * - **Timing Data Processing**: Converts streaming timing data to structured state
 * - **Error Handling**: Graceful handling when timing data is unavailable
 *
 * **Processing States:**
 * - `idle`: No active processing
 * - `generating`: Actively generating tokens
 *
 * **Token Rate Calculation:**
 * Uses timing data from `/chat/completions` streaming response for accurate
 * real-time token generation rate measurement.
 */
export class SlotsService {
	private callbacks: Set<(state: ApiProcessingState | null) => void> = new Set();
	private isStreamingActive: boolean = false;
	private lastKnownState: ApiProcessingState | null = null;

	/**
	 * Start streaming session tracking
	 */
	startStreaming(): void {
		this.isStreamingActive = true;
	}

	/**
	 * Stop streaming session tracking
	 */
	stopStreaming(): void {
		this.isStreamingActive = false;
	}

	/**
	 * Clear the current processing state
	 * Used when switching to a conversation without timing data
	 */
	clearState(): void {
		this.lastKnownState = null;

		for (const callback of this.callbacks) {
			try {
				callback(null);
			} catch (error) {
				console.error('Error in clearState callback:', error);
			}
		}
	}

	/**
	 * Check if currently in a streaming session
	 */
	isStreaming(): boolean {
		return this.isStreamingActive;
	}

	/**
	 * @deprecated Polling is no longer used - timing data comes from ChatService streaming response
	 * This method logs a warning if called to help identify outdated usage
	 */
	fetchAndNotify(): void {
		console.warn(
			'SlotsService.fetchAndNotify() is deprecated - use timing data from ChatService instead'
		);
	}

	subscribe(callback: (state: ApiProcessingState | null) => void): () => void {
		this.callbacks.add(callback);

		if (this.lastKnownState) {
			callback(this.lastKnownState);
		}

		return () => {
			this.callbacks.delete(callback);
		};
	}

	/**
	 * Updates processing state with timing data from ChatService streaming response
	 */
	async updateFromTimingData(timingData: {
		prompt_n: number;
		predicted_n: number;
		predicted_per_second: number;
		cache_n: number;
		prompt_progress?: ChatMessagePromptProgress;
	}): Promise<void> {
		const processingState = await this.parseCompletionTimingData(timingData);

		// Only update if we successfully parsed the state
		if (processingState === null) {
			console.warn('Failed to parse timing data - skipping update');
			return;
		}

		this.lastKnownState = processingState;

		for (const callback of this.callbacks) {
			try {
				callback(processingState);
			} catch (error) {
				console.error('Error in timing callback:', error);
			}
		}
	}

	/**
	 * Gets context total from last known slots data or fetches from server
	 */
	private async getContextTotal(): Promise<number | null> {
		if (this.lastKnownState && this.lastKnownState.contextTotal > 0) {
			return this.lastKnownState.contextTotal;
		}

		try {
			const currentConfig = config();
			const apiKey = currentConfig.apiKey?.toString().trim();

			const response = await fetch(`./slots`, {
				headers: {
					...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
				}
			});
			if (response.ok) {
				const slotsData = await response.json();
				if (Array.isArray(slotsData) && slotsData.length > 0) {
					const slot = slotsData[0];
					if (slot.n_ctx && slot.n_ctx > 0) {
						return slot.n_ctx;
					}
				}
			}
		} catch (error) {
			console.warn('Failed to fetch context total from /slots:', error);
		}

		return 4096;
	}

	private async parseCompletionTimingData(
		timingData: Record<string, unknown>
	): Promise<ApiProcessingState | null> {
		const promptTokens = (timingData.prompt_n as number) || 0;
		const predictedTokens = (timingData.predicted_n as number) || 0;
		const tokensPerSecond = (timingData.predicted_per_second as number) || 0;
		const cacheTokens = (timingData.cache_n as number) || 0;
		const promptProgress = timingData.prompt_progress as
			| {
					total: number;
					cache: number;
					processed: number;
					time_ms: number;
			  }
			| undefined;

		const contextTotal = await this.getContextTotal();

		if (contextTotal === null) {
			console.warn('No context total available - cannot calculate processing state');
			return null;
		}

		const currentConfig = config();
		const outputTokensMax = currentConfig.max_tokens || -1;

		const contextUsed = promptTokens + cacheTokens + predictedTokens;
		const outputTokensUsed = predictedTokens;

		const progressPercent = promptProgress
			? Math.round((promptProgress.processed / promptProgress.total) * 100)
			: undefined;

		return {
			status: predictedTokens > 0 ? 'generating' : promptProgress ? 'preparing' : 'idle',
			tokensDecoded: predictedTokens,
			tokensRemaining: outputTokensMax - predictedTokens,
			contextUsed,
			contextTotal,
			outputTokensUsed,
			outputTokensMax,
			hasNextToken: predictedTokens > 0,
			tokensPerSecond,
			temperature: currentConfig.temperature ?? 0.8,
			topP: currentConfig.top_p ?? 0.95,
			speculative: false,
			progressPercent,
			promptTokens,
			cacheTokens
		};
	}

	/**
	 * Get current processing state
	 * Returns the last known state from timing data, or null if no data available
	 */
	async getCurrentState(): Promise<ApiProcessingState | null> {
		if (this.lastKnownState) {
			return this.lastKnownState;
		}
		try {
			// Import dynamically to avoid circular dependency
			const { chatStore } = await import('$lib/stores/chat.svelte');
			const messages = chatStore.activeMessages;

			for (let i = messages.length - 1; i >= 0; i--) {
				const message = messages[i];
				if (message.role === 'assistant' && message.timings) {
					const restoredState = await this.parseCompletionTimingData({
						prompt_n: message.timings.prompt_n || 0,
						predicted_n: message.timings.predicted_n || 0,
						predicted_per_second:
							message.timings.predicted_n && message.timings.predicted_ms
								? (message.timings.predicted_n / message.timings.predicted_ms) * 1000
								: 0,
						cache_n: message.timings.cache_n || 0
					});

					if (restoredState) {
						this.lastKnownState = restoredState;
						return restoredState;
					}
				}
			}
		} catch (error) {
			console.warn('Failed to restore timing data from messages:', error);
		}

		return null;
	}
}

export const slotsService = new SlotsService();
