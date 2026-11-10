// Model UI's formatting and sizing helpers, from ui/apps/chat's HuggingFaceService (formatFileSize, formatSizeRange,
// getAvatarUrl) and utils (formatParameters, minMemoryTierGb). Pure functions: no fetching, no stores.
import {
	BYTE_LABEL,
	GIGABYTE,
	GIGABYTE_LABEL,
	HF_AVATARS_URL,
	KILOBYTE,
	KILOBYTE_LABEL,
	MB_PER_GB,
	MEGABYTE,
	MEGABYTE_LABEL,
	MEM_TIERS,
	MIB_BYTES,
	QUANT_WEIGHT,
	RAM_BUDGET_RATIO,
	RAM_OVERHEAD_MB
} from './constants.js';

/** A file size with one decimal, e.g. `4.6 GB`. */
export function formatFileSize(bytes: number): string {
	if (bytes >= GIGABYTE) return `${(bytes / GIGABYTE).toFixed(1)} ${GIGABYTE_LABEL}`;

	if (bytes >= MEGABYTE) return `${(bytes / MEGABYTE).toFixed(1)} ${MEGABYTE_LABEL}`;

	if (bytes >= KILOBYTE) return `${(bytes / KILOBYTE).toFixed(1)} ${KILOBYTE_LABEL}`;

	return `${bytes} ${BYTE_LABEL}`;
}

/** A min-max size range with one shared unit, e.g. `19.0-28.6 GB`. */
export function formatSizeRange(min: number, max: number): string {
	const unit =
		max >= GIGABYTE
			? GIGABYTE_LABEL
			: max >= MEGABYTE
				? MEGABYTE_LABEL
				: max >= KILOBYTE
					? KILOBYTE_LABEL
					: BYTE_LABEL;
	const div =
		unit === GIGABYTE_LABEL
			? GIGABYTE
			: unit === MEGABYTE_LABEL
				? MEGABYTE
				: unit === KILOBYTE_LABEL
					? KILOBYTE
					: 1;
	const fmt = (n: number) => (div === 1 ? `${n}` : `${(n / div).toFixed(1)}`);

	return `${fmt(min)}-${fmt(max)} ${unit}`;
}

/** A parameter or token count, e.g. `8B`, `262.14K`. */
export function formatParameters(params: number | unknown): string {
	if (typeof params !== 'number') return 'Unknown';

	if (params >= 1e9) {
		const billions = params / 1e9;

		return `${Number.isInteger(billions) ? billions : billions.toFixed(1)}B`;
	}

	if (params >= 1e6) return `${(params / 1e6).toFixed(2)}M`;

	if (params >= 1e3) return `${(params / 1e3).toFixed(2)}K`;

	return params.toString();
}

/**
 * Avatar URL for a Hugging Face author (org or user). 404s when the author does not exist, so callers should provide
 * a fallback, as ModelOrgAvatar does.
 */
export function getAvatarUrl(author: string): string {
	// OpenRouter-style model ids prefix the provider with a tilde (`~openai/gpt-...`); the avatars endpoint only
	// resolves bare names
	return `${HF_AVATARS_URL}/${author.replace(/^~/, '')}`;
}

/**
 * Smallest memory tier (GB) that can run a model of the given file size, or null if nothing fits even the largest
 * tier. `tiers` swaps in a platform's own sizes, such as the memory Macs ship with.
 */
export function minMemoryTierGb(sizeBytes: number, tiers: readonly number[] = MEM_TIERS): number | null {
	if (!sizeBytes) return null;

	const weightMb = (sizeBytes / MIB_BYTES) * QUANT_WEIGHT;

	for (const tier of tiers) {
		const budgetMb = tier * MB_PER_GB * RAM_BUDGET_RATIO - RAM_OVERHEAD_MB;

		if (weightMb <= budgetMb) return tier;
	}

	return null;
}

/**
 * A family's sizes in one line: the smallest to the largest and how many there are, e.g. `E2B–31B · 5 sizes`, or the
 * one size, e.g. `740M`. `params` lists one parameter count per size, smallest first; repeats (a size's instruct and
 * reasoning variants) count as sizes but appear once in the span.
 */
export function summarizeSizes(params: string[]): string {
	const unique = [...new Set(params)];

	if (params.length <= 1) return unique[0] ?? '';

	const span = unique.length === 1 ? unique[0] : `${unique[0]}–${unique[unique.length - 1]}`;

	return `${span} · ${params.length} sizes`;
}
