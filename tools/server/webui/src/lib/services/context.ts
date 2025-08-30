import { slotsService } from './slots';

export interface ContextCheckResult {
	wouldExceed: boolean;
	currentUsage: number;
	maxContext: number;
	availableTokens: number;
	reservedTokens: number;
}

/**
 * ContextService - Context window management and limit checking
 *
 * This service provides context window monitoring and limit checking using real-time
 * server data from the slots service. It helps prevent context overflow by tracking
 * current usage and calculating available space for new content.
 *
 * **Architecture & Relationships:**
 * - **ContextService** (this class): Context limit monitoring
 *   - Uses SlotsService for real-time context usage data
 *   - Calculates available tokens with configurable reserves
 *   - Provides context limit checking and error messaging
 *   - Helps prevent context window overflow
 *
 * - **SlotsService**: Provides current context usage from server slots
 * - **ChatStore**: Uses context checking before sending messages
 * - **UI Components**: Display context usage warnings and limits
 *
 * **Key Features:**
 * - **Real-time Context Checking**: Uses live server data for accuracy
 * - **Token Reservation**: Reserves tokens for response generation
 * - **Limit Detection**: Prevents context window overflow
 * - **Usage Reporting**: Detailed context usage statistics
 * - **Error Messaging**: User-friendly context limit messages
 * - **Configurable Reserves**: Adjustable token reservation for responses
 *
 * **Context Management:**
 * - Monitors current context usage from active slots
 * - Calculates available space considering reserved tokens
 * - Provides early warning before context limits are reached
 * - Helps optimize conversation length and content
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
