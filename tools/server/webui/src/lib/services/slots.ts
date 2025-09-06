import type { ApiProcessingState } from '$lib/types/api';
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
	private callbacks: Set<(state: ApiProcessingState) => void> = new Set();
	private isStreamingActive: boolean = false;
	private lastKnownState: ApiProcessingState | null = null;

	/**
	 * Start streaming session tracking
	 */
	startStreaming(): void {
		this.isStreamingActive = true;
	}

	/**
	 * Stop streaming session tracking and clear state
	 */
	stopStreaming(): void {
		this.isStreamingActive = false;
		this.lastKnownState = null;
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

	subscribe(callback: (state: ApiProcessingState) => void): () => void {
		this.callbacks.add(callback);
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
		// Return cached value if available
		if (this.lastKnownState && this.lastKnownState.contextTotal > 0) {
			return this.lastKnownState.contextTotal;
		}

		// Try to fetch context total from /slots endpoint
		try {
			const response = await fetch('/slots');
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

		// Fallback to reasonable default if no context data available
		return 4096;
	}

	private async parseCompletionTimingData(
		timingData: Record<string, unknown>
	): Promise<ApiProcessingState | null> {
		// Extract timing information from /chat/completions response
		const promptTokens = (timingData.prompt_n as number) || 0;
		const predictedTokens = (timingData.predicted_n as number) || 0;
		const tokensPerSecond = (timingData.predicted_per_second as number) || 0;

		// Get context total from server or cache
		const contextTotal = await this.getContextTotal();

		if (contextTotal === null) {
			console.warn('No context total available - cannot calculate processing state');
			return null;
		}

		// Get output max tokens from user settings
		const currentConfig = config();
		// Default to -1 (infinite) if max_tokens is not set, matching ChatService behavior
		const outputTokensMax = currentConfig.max_tokens || -1;

		// Calculate context and output based on team requirements
		const contextUsed = promptTokens + predictedTokens; // tokens_used_in_conversation
		const outputTokensUsed = predictedTokens; // tokens_generated_in_current_response

		return {
			status: predictedTokens > 0 ? 'generating' : 'idle',
			tokensDecoded: predictedTokens,
			tokensRemaining: outputTokensMax - predictedTokens,
			contextUsed,
			contextTotal,
			outputTokensUsed,
			outputTokensMax,
			hasNextToken: predictedTokens > 0,
			tokensPerSecond,
			// Use actual config values or reasonable defaults
			temperature: currentConfig.temperature ?? 0.8,
			topP: currentConfig.top_p ?? 0.95,
			speculative: false
		};
	}

	/**
	 * Get current processing state
	 * Returns the last known state from timing data, or null if no data available
	 */
	async getCurrentState(): Promise<ApiProcessingState | null> {
		return this.lastKnownState;
	}
}

export const slotsService = new SlotsService();
