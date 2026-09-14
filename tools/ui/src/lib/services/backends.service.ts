/**
 * BackendsService - Stateless backend connectivity checks
 *
 * Probes a backend's models endpoint to validate its URL and credentials.
 * No reactive state; consumed by the backends settings UI.
 */

import type { Backend } from '$lib/types';
import { isAbortError } from '$lib/utils/abort';
import { getAuthHeadersForBackend } from '$lib/utils/api-headers';
import { backendModelsUrl } from '$lib/utils/backend';

/** Outcome of a backend connectivity check. */
export interface BackendTestResult {
	error?: string;
	modelCount?: number;
	ok: boolean;
	status: number | null;
}

export class BackendsService {
	/**
	 * Check that a backend answers on its models endpoint.
	 *
	 * @param backend - Backend to probe. Does not need to be registered yet.
	 * @param signal - Optional abort signal for a cancelled test.
	 */
	static async test(backend: Backend, signal?: AbortSignal): Promise<BackendTestResult> {
		if (!backend.baseUrl.trim()) {
			return { error: 'Backend URL is required', ok: false, status: null };
		}

		try {
			const response = await fetch(backendModelsUrl(backend), {
				headers: getAuthHeadersForBackend(backend),
				signal
			});

			if (!response.ok) {
				return { error: await describeFailure(response), ok: false, status: response.status };
			}

			const body = (await response.json()) as { data?: unknown };
			const modelCount = Array.isArray(body?.data) ? body.data.length : 0;

			return { modelCount, ok: true, status: response.status };
		} catch (error) {
			if (isAbortError(error)) {
				return { ok: false, status: null };
			}

			return {
				error: error instanceof Error ? error.message : String(error),
				ok: false,
				status: null
			};
		}
	}
}

/** Build a human-readable message from a non-OK response. */
async function describeFailure(response: Response): Promise<string> {
	const status = `${response.status} ${response.statusText}`.trim();

	try {
		const body = (await response.json()) as { error?: { message?: string }; message?: string };
		const message = body?.error?.message ?? body?.message;

		if (message) return `${status}: ${message}`;
	} catch {
		// non-JSON error body, fall back to the status line
	}

	return status;
}
