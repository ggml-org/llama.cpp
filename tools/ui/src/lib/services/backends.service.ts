/**
 * BackendsService - Stateless backend connectivity checks and model listing
 *
 * Probes a backend's models endpoint to validate its URL and credentials, and
 * normalizes the response into the UI model shape. No reactive state;
 * consumed by the backends settings UI and the per-backend model cache.
 */

import { API_MODELS, LOCAL_BACKEND_ID } from '$lib/constants';
import type { ApiModelsListResponse, Backend, ModelOption } from '$lib/types';
import { isAbortError } from '$lib/utils/abort';
import { apiUrl } from '$lib/utils/api-base';
import { getAuthHeadersForBackend } from '$lib/utils/api-headers';
import { backendModelsUrl, readModelContextLength } from '$lib/utils/backend';

/** Models returned by a backend, plus the failure detail when the call fails. */
export interface BackendModelsResult {
	error?: string;
	models: ModelOption[];
	ok: boolean;
	status: number | null;
	/** Untouched list payload of the local backend, kept so the router rows and their load statuses can be rebuilt without asking again. */
	raw?: ApiModelsListResponse;
}

/** Outcome of a backend connectivity check. */
export interface BackendTestResult {
	error?: string;
	modelCount?: number;
	ok: boolean;
	status: number | null;
}

export class BackendsService {
	/**
	 * List the models a backend exposes on its models endpoint.
	 *
	 * @param backend - Backend to query. Does not need to be registered yet.
	 * @param signal - Optional abort signal for a cancelled request.
	 */
	static async listModels(backend: Backend, signal?: AbortSignal): Promise<BackendModelsResult> {
		// the local backend has no base URL; its models endpoint is base relative
		const url = backend.baseUrl.trim()
			? backendModelsUrl(backend)
			: apiUrl(API_MODELS.LIST, LOCAL_BACKEND_ID);

		if (!backend.baseUrl.trim() && backend.id !== LOCAL_BACKEND_ID) {
			return { error: 'Backend URL is required', models: [], ok: false, status: null };
		}

		try {
			const response = await fetch(url, {
				headers: getAuthHeadersForBackend(backend),
				signal
			});

			if (!response.ok) {
				return {
					error: await describeFailure(response),
					models: [],
					ok: false,
					status: response.status
				};
			}

			const body = (await response.json()) as { data?: unknown };
			const entries = Array.isArray(body?.data) ? body.data : [];
			const models = entries.flatMap((entry) => normalizeBackendModel(entry));
			// the local rows carry load status, external ones carry the context size
			const raw = body as ApiModelsListResponse;

			return { models, ok: true, raw, status: response.status };
		} catch (error) {
			if (isAbortError(error)) {
				return { models: [], ok: false, status: null };
			}

			return {
				error: error instanceof Error ? error.message : String(error),
				models: [],
				ok: false,
				status: null
			};
		}
	}

	/**
	 * Check that a backend answers on its models endpoint.
	 *
	 * @param backend - Backend to probe. Does not need to be registered yet.
	 * @param signal - Optional abort signal for a cancelled test.
	 */
	static async test(backend: Backend, signal?: AbortSignal): Promise<BackendTestResult> {
		const result = await BackendsService.listModels(backend, signal);

		return {
			error: result.error,
			modelCount: result.models.length,
			ok: result.ok,
			status: result.status
		};
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

/**
 * Normalize one entry of an OpenAI-compatible `/v1/models` response. External
 * backends only guarantee an id, so that doubles as the display name.
 */
function normalizeBackendModel(entry: unknown): ModelOption[] {
	if (!entry || typeof entry !== 'object') return [];

	const raw = entry as Record<string, unknown>;
	const id = typeof raw.id === 'string' ? raw.id.trim() : '';

	if (!id) return [];

	return [
		{ capabilities: [], contextLength: readModelContextLength(raw), id, model: id, name: id }
	];
}
