import { getBackend } from './api-base';
import { redactValue } from './redact';
import { ANTHROPIC_API_VERSION, CORS_PROXY, HEADERS } from '$lib/constants';
import { MimeTypeApplication } from '$lib/enums';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { Backend } from '$lib/types';

/**
 * Get authorization headers for API requests to a backend.
 */
export function getAuthHeaders(backendId?: string): Record<string, string> {
	const backend = getBackend(backendId);

	if (backend) return getAuthHeadersForBackend(backend);

	// no backends resolver yet (early startup, or a non-browser call): keep the
	// pre-backends behaviour and authenticate against the serving origin
	const apiKey = settingsStore.config.apiKey?.toString().trim();

	return apiKey ? { [HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${apiKey}` } : {};
}

/**
 * Get authorization headers for a backend object, including one that is not
 * registered yet (used by the connection test on the add-backend form).
 * Anthropic-compatible backends authenticate with x-api-key instead of Bearer.
 */
export function getAuthHeadersForBackend(backend: Backend): Record<string, string> {
	const headers: Record<string, string> = { ...(backend.headers ?? {}) };
	const apiKey = backend.apiKey?.trim();

	if (!apiKey) return headers;

	if (backend.protocol === 'anthropic') {
		headers[HEADERS.ANTHROPIC_API_KEY] = apiKey;
		headers[HEADERS.ANTHROPIC_VERSION] = ANTHROPIC_API_VERSION;
	} else {
		headers[HEADERS.AUTHORIZATION] = `${HEADERS.BEARER}${apiKey}`;
	}

	return headers;
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
