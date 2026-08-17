/**
 * Reasoning effort levels for thinking models.
 * These values are sent to the server and mapped to token budgets.
 */
export enum ReasoningEffort {
	DEFAULT = 'default',
	HIGH = 'high',
	LOW = 'low',
	MAX = 'max',
	MEDIUM = 'medium',
	OFF = 'off'
}
