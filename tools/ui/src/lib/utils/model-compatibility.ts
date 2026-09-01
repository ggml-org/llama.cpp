/**
 * Model memory estimation.
 *
 * Mirrors the app's compatibility check (Model+Compatibility.swift): the
 * runtime budget is RAM x 0.75 minus a fixed overhead, and a file fits when
 * its size with headroom stays under that budget. The result is the smallest
 * real Mac memory tier that can run the model, so the UI presents an honest
 * machine requirement instead of a raw file size. Context length and
 * device-specific budgets are deliberately ignored - callers present the
 * requirement and let the user judge.
 */
// LLAMA-APP-REUSE: hardware compatibility estimation

const MIB_BYTES = 1_048_576;
const MB_PER_GB = 1024;
/** Overhead multiplier applied to the file size when estimating weight memory. */
const QUANT_WEIGHT = 1.05;
/** Share of RAM the app allows the model to occupy. */
const RAM_BUDGET_RATIO = 0.75;
/** Fixed RAM overhead (MB) reserved for the system and KV cache. */
const RAM_OVERHEAD_MB = 2048;
/**
 * Memory tiers Macs ship with (GB). Tiers past 512 extrapolate Apple's step
 * pattern so builds too big for any current Mac still show an honest
 * requirement instead of silently omitting the line.
 */
const MAC_MEM_TIERS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 768, 1024];

/**
 * Estimated runtime memory (bytes) for a model of the given file size:
 * file size with headroom for KV cache and allocator overhead.
 */
export function estimateModelMemoryBytes(sizeBytes: number): number {
	return Math.round(sizeBytes * QUANT_WEIGHT);
}

/**
 * Smallest Mac memory tier (GB) that can run a model of the given file size,
 * or null if nothing fits even the largest tier.
 */
export function minMemoryTierGb(sizeBytes: number): number | null {
	if (!sizeBytes) return null;

	const weightMb = (sizeBytes / MIB_BYTES) * QUANT_WEIGHT;

	for (const tier of MAC_MEM_TIERS) {
		const budgetMb = tier * MB_PER_GB * RAM_BUDGET_RATIO - RAM_OVERHEAD_MB;

		if (weightMb <= budgetMb) return tier;
	}

	return null;
}
