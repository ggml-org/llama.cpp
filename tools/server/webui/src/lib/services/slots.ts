import type { ApiSlotData, ApiProcessingState } from '$lib/types/api';

export class SlotsService {
	private baseUrl: string;
	private pollingInterval: number;
	private pollingTimer: number | null = null;
	private callbacks: Set<(state: ApiProcessingState) => void> = new Set();

	constructor(baseUrl = '', pollingInterval = 500) {
		this.baseUrl = baseUrl;
		this.pollingInterval = pollingInterval;
	}

	startPolling(): void {
		if (this.pollingTimer) {
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
			const response = await fetch(`${this.baseUrl}/slots`);
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
		try {
			const response = await fetch(`${this.baseUrl}/slots`);
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
