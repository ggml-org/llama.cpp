import { slotsService } from './slots';

export interface ContextCheckResult {
	wouldExceed: boolean;
	currentUsage: number;
	maxContext: number;
	availableTokens: number;
	reservedTokens: number;
}

/**
 * Enhanced context service that uses real-time slots data for accurate context checking
 */
export class ContextService {
	private reserveTokens: number;

	constructor(reserveTokens = 512) {
		this.reserveTokens = reserveTokens;
	}

	/**
	 * Checks if the context limit would be exceeded
	 * 
	 * @returns {Promise<ContextCheckResult | null>} Promise that resolves to the context check result or null if an error occurs
	 */
	async checkContextLimit(): Promise<ContextCheckResult | null> {
		try {
			const currentState = await slotsService.getCurrentState();

			if (!currentState) {
				return null;
			}

			const maxContext = currentState.contextTotal;
			const currentUsage = currentState.contextUsed;
			const availableTokens = maxContext - currentUsage - this.reserveTokens;
			const wouldExceed = availableTokens <= 0;

			return {
				wouldExceed,
				currentUsage,
				maxContext,
				availableTokens: Math.max(0, availableTokens),
				reservedTokens: this.reserveTokens
			};
		} catch (error) {
			console.warn('Error checking context limit:', error);
			return null;
		}
	}

	/**
	 * Returns a formatted error message for context limit exceeded
	 * 
	 * @param {ContextCheckResult} result - Context check result
	 * @returns {string} Formatted error message
	 */
	getContextErrorMessage(result: ContextCheckResult): string {
		const usagePercent = Math.round((result.currentUsage / result.maxContext) * 100);
		return `Context window is nearly full. Current usage: ${result.currentUsage.toLocaleString()}/${result.maxContext.toLocaleString()} tokens (${usagePercent}%). Available space: ${result.availableTokens.toLocaleString()} tokens (${result.reservedTokens} reserved for response).`;
	}

	/**
	 * Sets the number of tokens to reserve for response generation
	 * 
	 * @param {number} tokens - Number of tokens to reserve
	 */
	setReserveTokens(tokens: number): void {
		this.reserveTokens = tokens;
	}
}

export const contextService = new ContextService();
