import type { ApiSlotData, ApiProcessingState } from '$lib/types/api';
import { slotsEndpointAvailable } from '$lib/stores/server.svelte';
import { SLOTS_DEBOUNCE_INTERVAL } from '$lib/constants/debounce';

/**
 * SlotsService - Real-time processing state monitoring and token rate calculation
 *
 * This service monitors the llama.cpp server's processing slots to provide real-time
 * information about generation progress, token rates, and context usage. It manages
 * streaming session tracking and provides debounced updates to prevent excessive
 * API calls during high-frequency streaming.
 *
 * **Architecture & Relationships:**
 * - **SlotsService** (this class): Processing state monitoring
 *   - Polls `/slots` endpoint for real-time server state
 *   - Calculates token generation rates and context usage
 *   - Manages streaming session lifecycle
 *   - Provides debounced updates during streaming
 *
 * - **ChatStore**: Uses slots service for streaming progress tracking
 * - **ServerStore**: Provides slots endpoint availability detection
 * - **UI Components**: Subscribe to processing state for progress indicators
 *
 * **Key Features:**
 * - **Real-time Monitoring**: Live processing state during generation
 * - **Token Rate Calculation**: Accurate tokens/second measurement
 * - **Context Tracking**: Current context usage and remaining capacity
 * - **Streaming Lifecycle**: Start/stop tracking for streaming sessions
 * - **Debounced Updates**: Prevents excessive API calls during streaming
 * - **State Parsing**: Converts raw slot data to structured processing state
 * - **Error Handling**: Graceful handling of endpoint unavailability
 *
 * **Processing States:**
 * - `idle`: No active processing
 * - `initializing`: Setting up generation context
 * - `preparing`: Preparing for token generation
 * - `generating`: Actively generating tokens
 *
 * **Token Rate Calculation:**
 * Uses both recent interval and total stream time for accurate rate measurement,
 * with moving average smoothing for stable display values.
 */
export class SlotsService {
	private callbacks: Set<(state: ApiProcessingState) => void> = new Set();
	private lastTokenCount: number = 0;
	private lastTimestamp: number = 0;
	private isStreamingActive: boolean = false;
	private currentTokensPerSecond: number = 0;
	private tokenRateHistory: number[] = [];
	private lastUpdateTime: number = 0;
	private streamStartTime: number = 0;
	private streamStartTokens: number = 0;
	private debounceTimer: ReturnType<typeof setTimeout> | null = null;
	private lastKnownState: ApiProcessingState | null = null;

	/**
	 * Start streaming session tracking
	 */
	startStreamingPolling(): void {
		this.isStreamingActive = true;
		this.streamStartTime = Date.now();
		this.streamStartTokens = 0;
		this.currentTokensPerSecond = 0;
		this.tokenRateHistory = [];
	}

	/**
	 * Stop streaming session tracking
	 */
	stopStreamingPolling(): void {
		this.isStreamingActive = false;
		this.lastTokenCount = 0;
		this.lastTimestamp = 0;
		this.currentTokensPerSecond = 0;
		this.tokenRateHistory = [];
		this.lastUpdateTime = 0;
		this.streamStartTime = 0;
		this.streamStartTokens = 0;

		if (this.debounceTimer !== null) {
			clearTimeout(this.debounceTimer);
			this.debounceTimer = null;
		}
	}

	/**
	 * Check if currently in a streaming session
	 */
	isStreaming(): boolean {
		return this.isStreamingActive;
	}

	/**
	 * Fetch and update slots state on demand (called during streaming chunks)
	 * Debounced to prevent excessive requests during high-frequency streaming
	 */
	async updateSlotsState(): Promise<void> {
		if (!this.isStreamingActive) {
			return;
		}

		const currentTime = Date.now();
		const timeSinceLastUpdate = currentTime - this.lastUpdateTime;

		if (timeSinceLastUpdate >= SLOTS_DEBOUNCE_INTERVAL) {
			if (this.debounceTimer !== null) {
				clearTimeout(this.debounceTimer);
				this.debounceTimer = null;
			}

			this.lastUpdateTime = currentTime;

			await this.performUpdate();
			return;
		}

		if (this.debounceTimer !== null) {
			return;
		}

		const waitTime = SLOTS_DEBOUNCE_INTERVAL - timeSinceLastUpdate;

		this.debounceTimer = setTimeout(async () => {
			this.debounceTimer = null;

			if (this.isStreamingActive) {
				this.lastUpdateTime = Date.now();
				await this.performUpdate();
			}
		}, waitTime);
	}

	/**
	 * Perform the actual slots state update
	 */
	private async performUpdate(): Promise<void> {
		const isAvailable = slotsEndpointAvailable();

		if (!isAvailable) {
			return;
		}

		if (!this.isStreamingActive) {
			return;
		}

		this.lastUpdateTime = Date.now();
		await this.fetchAndNotify();
	}

	subscribe(callback: (state: ApiProcessingState) => void): () => void {
		this.callbacks.add(callback);
		return () => {
			this.callbacks.delete(callback);
		};
	}

	private async fetchAndNotify(): Promise<void> {
		try {
			const response = await fetch(`/slots`);

			if (response.status === 501) {
				console.info('Slots endpoint not implemented');
				return;
			}

			if (!response.ok) {
				console.warn('Failed to fetch slots data:', response.statusText);
				return;
			}

			const slots: ApiSlotData[] = await response.json();
			const processingState = this.parseProcessingState(slots);

			this.lastKnownState = processingState;

			for (const callback of this.callbacks) {
				try {
					callback(processingState);
				} catch (error) {
					console.error('Error in slots callback:', error);
				}
			}
		} catch (error) {
			console.warn('Error fetching slots:', error);
		}
	}

	private parseProcessingState(slots: ApiSlotData[]): ApiProcessingState {
		const activeSlot = slots.find((slot) => slot.id_task !== -1) || slots[0];

		if (!activeSlot) {
			return {
				status: 'idle',
				tokensDecoded: 0,
				tokensRemaining: 0,
				contextUsed: 0,
				contextTotal: 4096,
				outputTokensUsed: 0,
				outputTokensMax: 2048,
				temperature: 0.8,
				topP: 0.95,
				speculative: false,
				hasNextToken: false,
				tokensPerSecond: 0
			};
		}

		let status: ApiProcessingState['status'] = 'idle';

		if (activeSlot.is_processing) {
			status = 'generating';
		} else if (activeSlot.next_token.n_decoded === 0 && activeSlot.id_task !== -1) {
			status = 'initializing';
		} else if (!activeSlot.next_token.has_next_token && activeSlot.id_task !== -1) {
			status = 'preparing';
		}

		// Calculate context and output token usage with the new slots format
		// n_decoded represents ALL tokens generated (thinking + regular content)
		const totalTokensGenerated = activeSlot.next_token.n_decoded;
		const maxOutputTokens = activeSlot.params.max_tokens || activeSlot.params.n_predict;
		
		// For context calculation: only count tokens that will be sent back to API
		// We need to estimate how many of the generated tokens are actual message content
		// vs thinking content. For now, we'll assume thinking is ~60% of total output
		// This is a rough estimate - in reality we'd need to track this separately
		const estimatedThinkingRatio = 0.6;
		const estimatedMessageTokens = Math.floor(totalTokensGenerated * (1 - estimatedThinkingRatio));
		
		// Context used = estimated prompt + only the message content tokens
		const maxGenerationTokens = Math.min(maxOutputTokens, Math.floor(activeSlot.n_ctx * 0.4));
		const estimatedPromptTokens = activeSlot.n_ctx - maxGenerationTokens;
		const contextUsed = Math.min(activeSlot.n_ctx, estimatedPromptTokens + estimatedMessageTokens);
		
		// Output tokens: total generated tokens (thinking + regular)
		const outputTokensUsed = totalTokensGenerated;
		const outputTokensMax = maxOutputTokens;

		const currentTime = Date.now();
		const currentTokens = activeSlot.next_token.n_decoded;

		if (this.isStreamingActive) {
			if (this.streamStartTokens === 0 && currentTokens > 0) {
				this.streamStartTokens = currentTokens;
				this.streamStartTime = currentTime;
			}

			let calculatedRate = 0;

			// Method 1: Use recent interval (preferred for accuracy)
			if (this.lastTimestamp > 0 && currentTokens > this.lastTokenCount) {
				const timeDiff = (currentTime - this.lastTimestamp) / 1000;
				const tokenDiff = currentTokens - this.lastTokenCount;

				if (timeDiff > 0.02) {
					calculatedRate = tokenDiff / timeDiff;
				}
			}

			// Method 2: Use total stream time (fallback for early display)
			if (
				calculatedRate === 0 &&
				this.streamStartTime > 0 &&
				currentTokens > this.streamStartTokens
			) {
				const totalTimeDiff = (currentTime - this.streamStartTime) / 1000;
				const totalTokenDiff = currentTokens - this.streamStartTokens;

				if (totalTimeDiff > 0.1) {
					// At least 100ms of streaming
					calculatedRate = totalTokenDiff / totalTimeDiff;
				}
			}

			if (calculatedRate > 0) {
				this.tokenRateHistory.push(calculatedRate);
				if (this.tokenRateHistory.length > 5) {
					this.tokenRateHistory.shift();
				}

				this.currentTokensPerSecond =
					this.tokenRateHistory.reduce((sum, rate) => sum + rate, 0) / this.tokenRateHistory.length;
			}
		}

		if (this.isStreamingActive && currentTokens >= this.lastTokenCount) {
			this.lastTokenCount = currentTokens;
			this.lastTimestamp = currentTime;
		}

		return {
			status,
			tokensDecoded: activeSlot.next_token.n_decoded,
			tokensRemaining: activeSlot.next_token.n_remain,
			contextUsed,
			contextTotal: activeSlot.n_ctx,
			outputTokensUsed,
			outputTokensMax,
			temperature: activeSlot.params.temperature,
			topP: activeSlot.params.top_p,
			speculative: activeSlot.speculative,
			hasNextToken: activeSlot.next_token.has_next_token,
			tokensPerSecond: this.currentTokensPerSecond
		};
	}

	async getCurrentState(): Promise<ApiProcessingState | null> {
		if (this.isStreamingActive) {
			return this.lastKnownState;
		}

		// For non-streaming state, check server store availability
		const isAvailable = slotsEndpointAvailable();

		if (!isAvailable) {
			return null;
		}

		try {
			const response = await fetch(`/slots`);

			if (response.status === 501) {
				console.info('Slots endpoint not implemented');
				return null;
			}

			if (!response.ok) {
				return null;
			}

			const slots: ApiSlotData[] = await response.json();
			return this.parseProcessingState(slots);
		} catch (error) {
			console.warn('Error fetching current slots state:', error);
			return null;
		}
	}
}

export const slotsService = new SlotsService();
