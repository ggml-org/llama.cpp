/**
 * Normalizes a value to a positive number, returning the fallback if invalid.
 * Handles both string and number inputs.
 */
export function normalizePositiveNumber(value: unknown, fallback: number): number {
	const parsed = typeof value === 'string' ? Number.parseFloat(value) : Number(value);
	if (!Number.isFinite(parsed) || parsed <= 0) {
		return fallback;
	}
	return parsed;
}
