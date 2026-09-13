/**
 * API base resolution for backends.
 *
 * The UI can talk to more than one backend endpoint. Services build request
 * URLs through {@link apiUrl} so a request always targets the right backend.
 * The backends store registers a resolver here; this module never imports the
 * store, which keeps URL resolution free of store dependencies.
 */

import { base } from '$app/paths';
import { API_ABSOLUTE_URL_PROTOCOLS } from '$lib/constants';
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
