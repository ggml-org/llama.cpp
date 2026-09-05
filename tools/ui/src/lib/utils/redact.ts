/**
 * Redacts a sensitive value, optionally showing the last N characters.
 *
 * @param value - The value to redact
 * @param showLastChars - If provided, reveals the last N characters with a leading mask
 * @returns The redacted string
 */
export function redactValue(value: string, showLastChars?: number): string {
	if (showLastChars) {
		return `....${value.slice(-showLastChars)}`;
	}

	return '[redacted]';
}
