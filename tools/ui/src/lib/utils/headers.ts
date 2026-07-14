/**
 * Header utilities for parsing and serializing HTTP headers.
 * Generic utilities not specific to MCP.
 */

/**
 * Parses a JSON string of headers into an array of key-value pairs.
 * Returns empty array if the JSON is invalid or empty.
 */
export function parseHeadersToArray(headersJson: string): { key: string; value: string }[] {
	if (!headersJson?.trim()) return [];

	try {
		const parsed = JSON.parse(headersJson);
		if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
			return Object.entries(parsed).map(([key, value]) => ({
				key,
				value: String(value)
			}));
		}
	} catch {
		return [];
	}

	return [];
}

/**
 * Serializes an array of header key-value pairs to a JSON string.
 * Filters out pairs with empty keys and returns empty string if no valid pairs.
 */
export function serializeHeaders(pairs: { key: string; value: string }[]): string {
	const validPairs = pairs.filter((p) => p.key.trim());

	if (validPairs.length === 0) return '';

	const obj: Record<string, string> = {};

	for (const pair of validPairs) {
		obj[pair.key.trim()] = pair.value;
	}

	return JSON.stringify(obj);
}
