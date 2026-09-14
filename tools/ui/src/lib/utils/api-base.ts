/**
 * API base resolution for backends.
 *
 * The UI can talk to more than one backend endpoint. Services build request
 * URLs through {@link apiUrl} so a request always targets the right backend.
 * The backends store registers a resolver here; this module never imports the
 * store, which keeps URL resolution free of store dependencies.
 */

import { backendChatUrl, backendModelsUrl } from './backend';
import { base } from '$app/paths';
import { API_ABSOLUTE_URL_PROTOCOLS, API_CHAT, API_MODELS } from '$lib/constants';
import type { Backend } from '$lib/types';

/** Backend list and active selection as exposed to URL resolution. */
export interface BackendsSnapshot {
	activeId: string;
	backends: Backend[];
}

type BackendsResolver = () => BackendsSnapshot;

let resolveBackends: BackendsResolver | null = null;

/** Registered once by the backends store. */
export function setBackendsResolver(resolver: BackendsResolver | null): void {
	resolveBackends = resolver;
}

/**
 * Look up a backend by id, defaulting to the active one.
 */
export function getBackend(backendId?: string): Backend | undefined {
	const snapshot = resolveBackends?.();

	if (!snapshot) return undefined;

	const id = backendId ?? snapshot.activeId;

	return snapshot.backends.find((backend) => backend.id === id);
}

/** API root for a backend, or an empty string for the local backend. */
export function getBackendBaseUrl(backendId?: string): string {
	return getBackend(backendId)?.baseUrl.trim() ?? '';
}

/**
 * Request target for a backend's chat completions endpoint. Returns a relative
 * path for the local backend and an absolute URL for external ones, so callers
 * can pass the result straight to `fetch` (or `apiFetch`, which resolves
 * relative paths against the base).
 */
export function apiChatUrl(backendId?: string): string {
	const backend = getBackend(backendId);

	if (backend?.baseUrl.trim()) {
		return backendChatUrl(backend);
	}

	return apiUrl(API_CHAT.COMPLETIONS, backendId);
}

/**
 * Request target for a backend's models listing. Returns a plain path for the
 * local backend (so `apiFetch` applies the base) and an absolute URL otherwise.
 */
export function apiModelsUrl(backendId?: string): string {
	const backend = getBackend(backendId);

	if (backend?.baseUrl.trim()) {
		return backendModelsUrl(backend);
	}

	return API_MODELS.LIST;
}

/**
 * Absolute URL for an API path on a backend.
 *
 * Absolute URLs pass through untouched. Paths on the local backend keep the
 * existing base-path-relative form, so serving under a subpath still works.
 * Paths on external backends resolve against the backend's API root.
 */
export function apiUrl(path: string, backendId?: string): string {
	if (API_ABSOLUTE_URL_PROTOCOLS.some((protocol) => path.startsWith(protocol))) {
		return path;
	}

	const baseUrl = getBackendBaseUrl(backendId);

	if (!baseUrl) {
		return `${base}${path}`;
	}

	const root = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;

	return new URL(path.replace(/^\.?\//, ''), root).toString();
}
