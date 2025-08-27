import type { ApiSlotData, ApiProcessingState } from '$lib/types/api';
import { serverStore } from '$lib/stores/server.svelte';

export class SlotsService {
	private callbacks: Set<(state: ApiProcessingState) => void> = new Set();
	private lastTokenCount: number = 0;
	private lastTimestamp: number = 0;
	private isStreamingActive: boolean = false;
	private currentTokensPerSecond: number = 0;
	private tokenRateHistory: number[] = [];
	private lastUpdateTime: number = 0;
	private pendingUpdate: boolean = false;
	private streamStartTime: number = 0;
	private streamStartTokens: number = 0;

	constructor() {}

	/**
	 * Check if slots endpoint is available based on server properties and endpoint support
	 */
	private async isSlotsEndpointAvailable(): Promise<boolean> {
		const serverProps = serverStore.serverProps;

		if (!serverProps) {
			return false;
		}

		if (serverProps.total_slots <= 0) {
			return false;
		}

		try {
			const response = await fetch('/slots');
			
			if (response.status === 501) {
				console.info('Slots endpoint not implemented - server started without --slots flag');
				return false;
			}
			
			return true;
		} catch (error) {
			console.warn('Unable to test slots endpoint availability:', error);
			return false;
		}
	}

	/**
	 * Reset slots availability check (call when server properties change)
	 */
	resetAvailabilityCheck(): void {
	}

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
		this.pendingUpdate = false;
		this.streamStartTime = 0;
		this.streamStartTokens = 0;
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

		// For the first few calls, use shorter debouncing to get tokens/sec faster
		const debounceTime = this.tokenRateHistory.length < 2 ? 50 : 100;

		if (timeSinceLastUpdate < debounceTime) {
			if (!this.pendingUpdate) {
				this.pendingUpdate = true;
				setTimeout(async () => {
					this.pendingUpdate = false;
					await this.performUpdate();
				}, debounceTime - timeSinceLastUpdate);
			}
			return;
		}

		await this.performUpdate();
	}


	/**
	 * Perform the actual slots state update
	 */
	private async performUpdate(): Promise<void> {
		if (!this.isStreamingActive) {
			return;
		}

		const isAvailable = await this.isSlotsEndpointAvailable();

		if (!isAvailable) {
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
			
			
			this.callbacks.forEach(callback => {
				try {
					callback(processingState);
				} catch (error) {
					console.error('Error in slots callback:', error);
				}
			});
		} catch (error) {
			console.warn('Error fetching slots:', error);
		}
	}

	private parseProcessingState(slots: ApiSlotData[]): ApiProcessingState {
		const activeSlot = slots.find(slot => slot.id_task !== -1) || slots[0];

		if (!activeSlot) {
			return {
				status: 'idle',
				tokensDecoded: 0,
				tokensRemaining: 0,
				contextUsed: 0,
				contextTotal: 4096,
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

		const promptTokens = Math.floor(activeSlot.prompt.length / 4);
		const contextUsed = promptTokens + activeSlot.next_token.n_decoded;

		const currentTime = Date.now();
		const currentTokens = activeSlot.next_token.n_decoded;
		
		if (this.isStreamingActive) {
			// Initialize stream tracking on first call
			if (this.streamStartTokens === 0 && currentTokens > 0) {
				this.streamStartTokens = currentTokens;
				this.streamStartTime = currentTime;
			}
			
			// Calculate tokens/sec using multiple methods for reliability
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
			if (calculatedRate === 0 && this.streamStartTime > 0 && currentTokens > this.streamStartTokens) {
				const totalTimeDiff = (currentTime - this.streamStartTime) / 1000;
				const totalTokenDiff = currentTokens - this.streamStartTokens;
				
				if (totalTimeDiff > 0.1) { // At least 100ms of streaming
					calculatedRate = totalTokenDiff / totalTimeDiff;
				}
			}
			
			// Update rate if we have a valid calculation
			if (calculatedRate > 0) {
				this.tokenRateHistory.push(calculatedRate);
				if (this.tokenRateHistory.length > 5) {
					this.tokenRateHistory.shift();
				}
				
				this.currentTokensPerSecond = this.tokenRateHistory.reduce((sum, rate) => sum + rate, 0) / this.tokenRateHistory.length;
			}
			
			// Always show some rate during active streaming (even if 0 initially)
			// This ensures the UI always displays tokens/sec field during streaming
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
			temperature: activeSlot.params.temperature,
			topP: activeSlot.params.top_p,
			speculative: activeSlot.speculative,
			hasNextToken: activeSlot.next_token.has_next_token,
			tokensPerSecond: this.currentTokensPerSecond
		};
	}

	async getCurrentState(): Promise<ApiProcessingState | null> {
		const isAvailable = await this.isSlotsEndpointAvailable();
		
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
