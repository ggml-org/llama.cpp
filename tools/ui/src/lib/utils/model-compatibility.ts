/**
 * Model memory estimation.
 *
 * Mirrors the app's compatibility check (Model+Compatibility.swift): the
 * runtime budget is RAM x 0.75 minus a fixed overhead, and a file fits when
 * its size with headroom stays under that budget. The result is the smallest
 * memory tier that can run the model, so the UI presents an honest machine
 * requirement instead of a raw file size. Context length and
 * device-specific budgets are deliberately ignored - callers present the
 * requirement and let the user judge.
 */
import {
	MB_PER_GB,
	MEM_TIERS,
	MIB_BYTES,
	QUANT_WEIGHT,
	RAM_BUDGET_RATIO,
	RAM_OVERHEAD_MB
} from '$lib/constants';

/**
 * Smallest memory tier (GB) that can run a model of the given file size,
 * or null if nothing fits even the largest tier.
 */
export function minMemoryTierGb(sizeBytes: number): number | null {
	if (!sizeBytes) return null;

	const weightMb = (sizeBytes / MIB_BYTES) * QUANT_WEIGHT;

	for (const tier of MEM_TIERS) {
		const budgetMb = tier * MB_PER_GB * RAM_BUDGET_RATIO - RAM_OVERHEAD_MB;

		if (weightMb <= budgetMb) return tier;
	}

	return null;
}
