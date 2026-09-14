/**
 * Backend list parsing, defaults and endpoint URLs.
 *
 * External backends are persisted in settings as a JSON list. Malformed
 * entries are dropped instead of throwing so a corrupted settings value can
 * never break URL resolution.
 */

import {
	BACKEND_CAPABILITIES,
	BACKEND_ID_PREFIX,
	BACKEND_PROTOCOLS,
	DEFAULT_BACKEND_CHAT_PATH,
	DEFAULT_BACKEND_MODELS_PATH,
	LOCAL_BACKEND_ID
} from '$lib/constants';
import type { Backend, BackendCapabilities, BackendProtocol } from '$lib/types';

/** Absolute chat completions URL for a backend. */
export function backendChatUrl(backend: Backend): string {
	return joinBackendUrl(backend.baseUrl, backend.chatPath ?? DEFAULT_BACKEND_CHAT_PATH);
}

/** Absolute models listing URL for a backend. */
export function backendModelsUrl(backend: Backend): string {
	return joinBackendUrl(backend.baseUrl, backend.modelsPath ?? DEFAULT_BACKEND_MODELS_PATH);
}

/** Features a backend supports, derived from its protocol. */
export function getBackendCapabilities(backend: Backend): BackendCapabilities {
	return BACKEND_CAPABILITIES[backend.protocol] ?? BACKEND_CAPABILITIES.openai;
}

/** The built-in backend pointing at the server that serves this UI. */
export function createLocalBackend(apiKey?: string, enabled = true): Backend {
	return {
		apiKey,
		baseUrl: '',
		enabled,
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

function joinBackendUrl(baseUrl: string, path: string): string {
	const base = baseUrl.replace(/\/+$/, '');
	const suffix = path.startsWith('/') ? path : `/${path}`;

	return `${base}${suffix}`;
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
		chatPath: parseOptionalPath(raw.chatPath),
		enabled: raw.enabled !== false,
		headers: parseBackendHeaders(raw.headers),
		id,
		modelsPath: parseOptionalPath(raw.modelsPath),
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

function parseOptionalPath(raw: unknown): string | undefined {
	return typeof raw === 'string' && raw.trim() ? raw.trim() : undefined;
}
