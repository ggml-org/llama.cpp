/**
 * Model memory-fit constants.
 *
 * Mirrors the app's compatibility check (Model+Compatibility.swift):
 *   budget      = RAM x RAM_BUDGET_RATIO - RAM_OVERHEAD_MB
 *   weightBytes = fileBytes x QUANT_WEIGHT
 * a file fits when weightBytes <= budget. Kept here so the estimation util and
 * any caller share one source.
 */

export {
	MB_PER_GB,
	MEM_TIERS,
	MIB_BYTES,
	QUANT_WEIGHT,
	RAM_BUDGET_RATIO,
	RAM_OVERHEAD_MB
} from '@ggml-org/llama-ui/models';
