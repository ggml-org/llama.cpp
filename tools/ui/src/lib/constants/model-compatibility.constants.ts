/**
 * Model memory-fit constants.
 *
 * Mirrors the app's compatibility check (Model+Compatibility.swift):
 *   budget      = RAM x RAM_BUDGET_RATIO - RAM_OVERHEAD_MB
 *   weightBytes = fileBytes x QUANT_WEIGHT
 * a file fits when weightBytes <= budget. Kept here so the estimation util and
 * any caller share one source.
 */

/** Bytes in one mebibyte (MiB), used to convert a file size to MB. */
export const MIB_BYTES = 1_048_576;

/** MB in one GB. */
export const MB_PER_GB = 1024;

/** Overhead multiplier applied to the file size when estimating weight memory. */
export const QUANT_WEIGHT = 1.05;

/** Share of RAM the app allows the model to occupy. */
export const RAM_BUDGET_RATIO = 0.75;

/** Fixed RAM overhead (MB) reserved for the system and KV cache. */
export const RAM_OVERHEAD_MB = 2048;

/**
 * Memory tiers Macs ship with (GB). Tiers past 512 extrapolate Apple's step
 * pattern so builds too big for any current Mac still show an honest
 * requirement instead of silently omitting the line.
 */
export const MAC_MEM_TIERS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 768, 1024];
