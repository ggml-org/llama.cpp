/**
 * Reasoning effort levels for thinking models.
 * A level the chat template honors is sent as reasoning_effort, the others map to token budgets.
 */
export enum ReasoningEffort {
	DEFAULT = 'default',
	HIGH = 'high',
	LOW = 'low',
	MAX = 'max',
	MEDIUM = 'medium',
	MINIMAL = 'minimal',
	OFF = 'off',
	XHIGH = 'xhigh'
}
