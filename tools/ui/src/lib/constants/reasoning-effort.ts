import { ReasoningEffort } from '$lib/enums';
import type { ReasoningEffortLevel } from '$lib/types';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const REASONING_EFFORT_LABELS: Record<string, string> = {
	[ReasoningEffort.DEFAULT]: 'Default',
	[ReasoningEffort.OFF]: 'Off',
	[ReasoningEffort.LOW]: 'Low',
	[ReasoningEffort.MEDIUM]: 'Medium',
	[ReasoningEffort.HIGH]: 'High',
	[ReasoningEffort.MAX]: 'Max'
};

export const REASONING_EFFORT_LEVELS: ReasoningEffortLevel[] = [
	{ value: ReasoningEffort.DEFAULT, label: 'Default' },
	{ value: ReasoningEffort.OFF, label: 'Off' },
	{ value: ReasoningEffort.LOW, label: 'Low' },
	{ value: ReasoningEffort.MEDIUM, label: 'Medium' },
	{ value: ReasoningEffort.HIGH, label: 'High' },
	{ value: ReasoningEffort.MAX, label: 'Max', hasInfo: true }
];
