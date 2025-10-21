/**
 * Normalizes a model name by extracting the filename from a path.
 *
 * Handles both forward slashes (/) and backslashes (\) as path separators.
 * If the model name is just a filename (no path), returns it as-is.
 *
 * @param modelName - The model name or path to normalize
 * @returns The normalized model name (filename only)
 */
export function normalizeModelName(modelName: string): string {
	const trimmed = modelName.trim();

	if (!trimmed) {
		return '';
	}

	const segments = trimmed.split(/[\\/]/);
	const candidate = segments.pop();
	const normalized = candidate?.trim();

	return normalized && normalized.length > 0 ? normalized : trimmed;
}

/**
 * Validates if a model name is valid (non-empty after normalization).
 *
 * @param modelName - The model name to validate
 * @returns true if valid, false otherwise
 */
export function isValidModelName(modelName: string): boolean {
	return normalizeModelName(modelName).length > 0;
}
