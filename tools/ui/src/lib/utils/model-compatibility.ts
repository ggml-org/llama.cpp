/**
 * Model memory estimation.
 *
 * Runtime memory is approximated from the file size: the quantized weights
 * plus KV cache/workspace overhead, rounded up to a GB. Context length and
 * device-specific budgets are deliberately ignored - callers present the
 * requirement and let the user judge.
 */

/** Overhead multiplier applied to the file size when estimating weight memory. */
const WEIGHT_OVERHEAD_MULTIPLIER = 1.05;

/**
 * Estimated runtime memory (bytes) for a model of the given file size:
 * file size with headroom for KV cache and allocator overhead.
 */
export function estimateModelMemoryBytes(sizeBytes: number): number {
	return Math.round(sizeBytes * WEIGHT_OVERHEAD_MULTIPLIER);
}
