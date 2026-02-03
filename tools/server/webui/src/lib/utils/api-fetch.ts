import { base } from '$app/paths';
import { getJsonHeaders, getAuthHeaders } from './api-headers';
import { UrlPrefix } from '$lib/enums';

/**
 * API Fetch Utilities
 *
 * Provides common fetch patterns used across services:
 * - Automatic JSON headers
 * - Error handling with proper error messages
 * - Base path resolution
 */

export interface ApiFetchOptions extends Omit<RequestInit, 'headers'> {
	/**
	 * Use auth-only headers (no Content-Type).
	 * Default: false (uses JSON headers with Content-Type: application/json)
	 */
	authOnly?: boolean;
	/**
	 * Additional headers to merge with default headers.
	 */
	headers?: Record<string, string>;
}

/**
 * Fetch JSON data from an API endpoint with standard headers and error handling.
 *
 * @param path - API path (will be prefixed with base path)
 * @param options - Fetch options with additional authOnly flag
 * @returns Parsed JSON response
 * @throws Error with formatted message on failure
 *
 * @example
 * ```typescript
 * // GET request
 * const models = await apiFetch<ApiModelListResponse>('/v1/models');
 *
 * // POST request
 * const result = await apiFetch<ApiResponse>('/models/load', {
 *   method: 'POST',
 *   body: JSON.stringify({ model: 'gpt-4' })
 * });
 * ```
 */
export async function apiFetch<T>(path: string, options: ApiFetchOptions = {}): Promise<T> {
	const { authOnly = false, headers: customHeaders, ...fetchOptions } = options;

	const baseHeaders = authOnly ? getAuthHeaders() : getJsonHeaders();
	const headers = { ...baseHeaders, ...customHeaders };

	const url =
		path.startsWith(UrlPrefix.HTTP) || path.startsWith(UrlPrefix.HTTPS) ? path : `${base}${path}`;

	const response = await fetch(url, {
		...fetchOptions,
		headers
	});

	if (!response.ok) {
		const errorMessage = await parseErrorMessage(response);
		throw new Error(errorMessage);
	}

	return response.json() as Promise<T>;
}

/**
 * Fetch with URL constructed from base URL and query parameters.
 *
 * @param basePath - Base API path
 * @param params - Query parameters to append
 * @param options - Fetch options
 * @returns Parsed JSON response
 *
 * @example
 * ```typescript
 * const props = await apiFetchWithParams<ApiProps>('./props', {
 *   model: 'gpt-4',
 *   autoload: 'false'
 * });
 * ```
 */
export async function apiFetchWithParams<T>(
	basePath: string,
	params: Record<string, string>,
	options: ApiFetchOptions = {}
): Promise<T> {
	const url = new URL(basePath, window.location.href);

	for (const [key, value] of Object.entries(params)) {
		if (value !== undefined && value !== null) {
			url.searchParams.set(key, value);
		}
	}

	const { authOnly = false, headers: customHeaders, ...fetchOptions } = options;

	const baseHeaders = authOnly ? getAuthHeaders() : getJsonHeaders();
	const headers = { ...baseHeaders, ...customHeaders };

	const response = await fetch(url.toString(), {
		...fetchOptions,
		headers
	});

	if (!response.ok) {
		const errorMessage = await parseErrorMessage(response);
		throw new Error(errorMessage);
	}

	return response.json() as Promise<T>;
}

/**
 * POST JSON data to an API endpoint.
 *
 * @param path - API path
 * @param body - Request body (will be JSON stringified)
 * @param options - Additional fetch options
 * @returns Parsed JSON response
 */
export async function apiPost<T, B = unknown>(
	path: string,
	body: B,
	options: ApiFetchOptions = {}
): Promise<T> {
	return apiFetch<T>(path, {
		method: 'POST',
		body: JSON.stringify(body),
		...options
	});
}

/**
 * Parse error message from a failed response.
 * Tries to extract error message from JSON body, falls back to status text.
 */
async function parseErrorMessage(response: Response): Promise<string> {
	try {
		const errorData = await response.json();
		if (errorData?.error?.message) {
			return errorData.error.message;
		}
		if (errorData?.error && typeof errorData.error === 'string') {
			return errorData.error;
		}
		if (errorData?.message) {
			return errorData.message;
		}
	} catch {
		// JSON parsing failed, use status text
	}

	return `Request failed: ${response.status} ${response.statusText}`;
}
