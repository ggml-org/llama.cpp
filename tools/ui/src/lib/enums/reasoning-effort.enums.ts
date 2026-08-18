/**
 * Reasoning effort levels for thinking models.
 * These values are sent to the server and mapped to token budgets.
 */
export enum ReasoningEffort {
	DEFAULT = 'default',
	OFF = 'off',
	LOW = 'low',
	MEDIUM = 'medium',
	HIGH = 'high',
	MAX = 'max'
}
