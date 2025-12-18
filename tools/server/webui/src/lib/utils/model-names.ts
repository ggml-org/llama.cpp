/**
 * Normalizes a model name by extracting the filename from a path, but preserves Hugging Face repository format.
 *
 * Handles both forward slashes (/) and backslashes (\) as path separators.
 * - If the model name has exactly one slash (org/model format), preserves the full "org/model" name
 * - If the model name has no slash or multiple slashes, extracts just the filename
 * - If the model name is just a filename (no path), returns it as-is.
 *
 * @param modelName - The model name or path to normalize
 * @returns The normalized model name
 *
 * @example
 * normalizeModelName('models/llama-3.1-8b') // Returns: 'llama-3.1-8b' (multiple slashes -> filename)
 * normalizeModelName('C:\\Models\\gpt-4') // Returns: 'gpt-4' (multiple slashes -> filename)
 * normalizeModelName('meta-llama/Llama-3.1-8B') // Returns: 'meta-llama/Llama-3.1-8B' (Hugging Face format)
 * normalizeModelName('simple-model') // Returns: 'simple-model' (no slash)
 * normalizeModelName('  spaced  ') // Returns: 'spaced'
 * normalizeModelName('') // Returns: ''
 */
export function normalizeModelName(modelName: string): string {
	const trimmed = modelName.trim();

	if (!trimmed) {
		return '';
	}

	const segments = trimmed.split(/[\\/]/);

	// If we have exactly 2 segments (one slash), treat it as Hugging Face repo format
	// and preserve the full "org/model" format
	if (segments.length === 2) {
		const [org, model] = segments;
		const trimmedOrg = org?.trim();
		const trimmedModel = model?.trim();

		if (trimmedOrg && trimmedModel) {
			return `${trimmedOrg}/${trimmedModel}`;
		}
	}

	// For other cases (no slash, or multiple slashes), extract just the filename
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
