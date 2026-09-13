/**
 * Backend list parsing and defaults.
 *
 * External backends are persisted in settings as a JSON list. Malformed
 * entries are dropped instead of throwing so a corrupted settings value can
 * never break URL resolution.
 */

import { BACKEND_ID_PREFIX, BACKEND_PROTOCOLS, LOCAL_BACKEND_ID } from '$lib/constants';
import type { Backend, BackendProtocol } from '$lib/types';

/** The built-in backend pointing at the server that serves this UI. */
export function createLocalBackend(): Backend {
	return {
		baseUrl: '',
		enabled: true,
		id: LOCAL_BACKEND_ID,
		name: 'Local',
		protocol: 'llama.cpp'
	};
}

/**
 * Parse the persisted backends JSON into backend entries.
 */
export function parseBackendsSettings(rawBackends: unknown): Backend[] {
	if (!rawBackends) return [];

	let parsed: unknown;

	if (typeof rawBackends === 'string') {
		const trimmed = rawBackends.trim();

		if (!trimmed) return [];

		try {
			parsed = JSON.parse(trimmed);
		} catch (error) {
			console.warn('[backends] Failed to parse backends JSON, ignoring value:', error);

			return [];
		}
	} else {
		parsed = rawBackends;
	}

	if (!Array.isArray(parsed)) return [];

	return parsed.flatMap((entry, index) => {
		const backend = parseBackendEntry(entry, index);

		return backend ? [backend] : [];
	});
}

function parseBackendEntry(entry: unknown, index: number): Backend | null {
	if (!entry || typeof entry !== 'object') return null;

	const raw = entry as Record<string, unknown>;
	const baseUrl = typeof raw.baseUrl === 'string' ? raw.baseUrl.trim() : '';

	// the local backend is built in and never persisted
	if (!baseUrl || raw.id === LOCAL_BACKEND_ID) return null;

	const protocol = BACKEND_PROTOCOLS.includes(raw.protocol as BackendProtocol)
		? (raw.protocol as BackendProtocol)
		: 'openai';
	const id =
		typeof raw.id === 'string' && raw.id.trim()
			? raw.id.trim()
			: `${BACKEND_ID_PREFIX}-${index + 1}`;
	const name = typeof raw.name === 'string' && raw.name.trim() ? raw.name.trim() : baseUrl;
	const apiKey =
		typeof raw.apiKey === 'string' && raw.apiKey.trim() ? raw.apiKey.trim() : undefined;

	return {
		apiKey,
		baseUrl,
		enabled: raw.enabled !== false,
		headers: parseBackendHeaders(raw.headers),
		id,
		name,
		protocol
	};
}

function parseBackendHeaders(raw: unknown): Record<string, string> | undefined {
	if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined;

	const entries = Object.entries(raw as Record<string, unknown>)
		.filter(([, value]) => typeof value === 'string' && value.trim() !== '')
		.map(([key, value]) => [key.trim(), (value as string).trim()] as const);

	return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}
