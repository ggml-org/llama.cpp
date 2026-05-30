import { ReasoningEffort } from '$lib/enums';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const REASONING_EFFORT_LABELS: Record<string, string> = {
	[ReasoningEffort.LOW]: 'Low',
	[ReasoningEffort.MEDIUM]: 'Medium',
	[ReasoningEffort.HIGH]: 'High',
	[ReasoningEffort.MAX]: 'Max'
};
