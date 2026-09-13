import { getBackend } from './api-base';
import { redactValue } from './redact';
import { CORS_PROXY, HEADERS, LOCAL_BACKEND_ID } from '$lib/constants';
import { MimeTypeApplication } from '$lib/enums';
import { settingsStore } from '$lib/stores/settings/index.svelte';

/**
 * Get authorization headers for API requests to a backend.
 * External backends carry their own key; the local backend reuses the global
 * API key setting.
 */
export function getAuthHeaders(backendId?: string): Record<string, string> {
	const apiKey = resolveBackendApiKey(backendId);

	return apiKey ? { [HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${apiKey}` } : {};
}

/**
 * Get standard JSON headers with optional authorization
 */
export function getJsonHeaders(backendId?: string): Record<string, string> {
	return {
		[HEADERS.CONTENT_TYPE]: MimeTypeApplication.JSON,
		...getAuthHeaders(backendId)
	};
}

function resolveBackendApiKey(backendId?: string): string | undefined {
	const backend = getBackend(backendId);
	const backendKey = backend?.apiKey?.trim();

	if (backendKey) return backendKey;

	// an external backend without its own key must not receive the local key
	if (backend && backend.id !== LOCAL_BACKEND_ID) return undefined;

	return settingsStore.config.apiKey?.toString().trim() || undefined;
}

/**
 * Sanitize HTTP headers by redacting sensitive values.
 * Known sensitive headers (from HEADERS.REDACTED) and any extra headers
 * specified by the caller are fully redacted. Headers listed in
 * `partialRedactHeaders` are partially redacted, showing only the
 * specified number of trailing characters.
 *
 * @param headers - Headers to sanitize
 * @param extraRedactedHeaders - Additional header names to fully redact
 * @param partialRedactHeaders - Map of header name -> number of trailing chars to keep visible
 * @returns Object with header names as keys and (possibly redacted) values
 */
export function sanitizeHeaders(
	headers?: HeadersInit,
	extraRedactedHeaders?: Iterable<string>,
	partialRedactHeaders?: Map<string, number>
): Record<string, string> {
	if (!headers) {
		return {};
	}

	const normalized = new Headers(headers);
	const sanitized: Record<string, string> = {};
	const redactedHeaders = new Set(
		Array.from(extraRedactedHeaders ?? [], (header) => header.toLowerCase())
	);

	for (const [key, value] of normalized.entries()) {
		const normalizedKey = key.toLowerCase();
		const unproxiedKey = normalizedKey.startsWith(CORS_PROXY.HEADER_PREFIX)
			? normalizedKey.slice(CORS_PROXY.HEADER_PREFIX.length)
			: normalizedKey;
		const partialChars =
			partialRedactHeaders?.get(normalizedKey) ?? partialRedactHeaders?.get(unproxiedKey);

		if (partialChars !== undefined) {
			sanitized[key] = redactValue(value, partialChars);
		} else if (
			HEADERS.REDACTED.has(normalizedKey) ||
			HEADERS.REDACTED.has(unproxiedKey) ||
			redactedHeaders.has(normalizedKey) ||
			redactedHeaders.has(unproxiedKey)
		) {
			sanitized[key] = redactValue(value);
		} else {
			sanitized[key] = value;
		}
	}

	return sanitized;
}
