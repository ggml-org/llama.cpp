import type { ApiSlotData, ApiProcessingState } from '$lib/types/api';
import { serverStore } from '$lib/stores/server.svelte';

export class SlotsService {
	private pollingInterval: number;
	private pollingTimer: number | null = null;
	private callbacks: Set<(state: ApiProcessingState) => void> = new Set();
	private slotsAvailable: boolean | null = null;
	private slotsEndpointSupported: boolean | null = null;

	constructor(pollingInterval = 500) {
		this.pollingInterval = pollingInterval;
	}

	/**
	 * Check if slots endpoint is available based on server properties and endpoint support
	 */
	private async isSlotsEndpointAvailable(): Promise<boolean> {
		// If we've already determined endpoint support, use cached result
		if (this.slotsEndpointSupported !== null) {
			return this.slotsEndpointSupported;
		}

		// First check server properties
		const serverProps = serverStore.serverProps;
		if (!serverProps) {
			this.slotsEndpointSupported = false;
			return false;
		}

		// Check if server has slots support (total_slots > 0)
		if (serverProps.total_slots <= 0) {
			this.slotsEndpointSupported = false;
			return false;
		}

		// Test if the endpoint is actually implemented
		try {
			const response = await fetch('/slots');
			
			// Handle 501 Not Implemented specifically
			if (response.status === 501) {
				console.info('Slots endpoint not implemented - server started without --slots flag');
				this.slotsEndpointSupported = false;
				return false;
			}
			
			// If we get any successful response or other error, assume it's supported
			this.slotsEndpointSupported = true;
			return true;
		} catch (error) {
			// Network errors - assume endpoint might be supported but server is down
			console.warn('Unable to test slots endpoint availability:', error);
			this.slotsEndpointSupported = false;
			return false;
		}
	}

	/**
	 * Reset slots availability check (call when server properties change)
	 */
	resetAvailabilityCheck(): void {
		this.slotsAvailable = null;
		this.slotsEndpointSupported = null;
	}

	async startPolling(): Promise<void> {
		if (this.pollingTimer) {
			return;
		}

		// Only start polling if slots endpoint is available
		const isAvailable = await this.isSlotsEndpointAvailable();
		if (!isAvailable) {
			console.info('Slots endpoint not available - polling disabled');
			return;
		}

		this.poll();
		this.pollingTimer = window.setInterval(() => {
			this.poll();
		}, this.pollingInterval);
	}

	stopPolling(): void {
		if (this.pollingTimer) {
			clearInterval(this.pollingTimer);
			this.pollingTimer = null;
		}
	}

	subscribe(callback: (state: ApiProcessingState) => void): () => void {
		this.callbacks.add(callback);
		return () => {
			this.callbacks.delete(callback);
		};
	}

	private async poll(): Promise<void> {
		try {
			const response = await fetch(`/slots`);
			
			// Handle 501 Not Implemented - stop polling and mark as unsupported
			if (response.status === 501) {
				console.info('Slots endpoint not implemented - stopping polling');
				this.slotsEndpointSupported = false;
				this.stopPolling();
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
			console.warn('Error polling slots:', error);
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
				hasNextToken: false
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

		// Calculate context usage (estimate based on prompt length and decoded tokens)
		const promptTokens = Math.floor(activeSlot.prompt.length / 4); // Rough estimate
		const contextUsed = promptTokens + activeSlot.next_token.n_decoded;

		return {
			status,
			tokensDecoded: activeSlot.next_token.n_decoded,
			tokensRemaining: activeSlot.next_token.n_remain,
			contextUsed,
			contextTotal: activeSlot.n_ctx,
			temperature: activeSlot.params.temperature,
			topP: activeSlot.params.top_p,
			speculative: activeSlot.speculative,
			hasNextToken: activeSlot.next_token.has_next_token
		};
	}

	async getCurrentState(): Promise<ApiProcessingState | null> {
		// Check if slots endpoint is available before making request
		const isAvailable = await this.isSlotsEndpointAvailable();
		if (!isAvailable) {
			return null;
		}

		try {
			const response = await fetch(`/slots`);
			
			// Handle 501 Not Implemented
			if (response.status === 501) {
				console.info('Slots endpoint not implemented');
				this.slotsEndpointSupported = false;
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
