/**
 * Backend list parsing, defaults and endpoint URLs.
 *
 * External backends are persisted in settings as a JSON list. Malformed
 * entries are dropped instead of throwing so a corrupted settings value can
 * never break URL resolution.
 */

import {
	BACKEND_CAPABILITIES,
	BACKEND_COMPAT,
	BACKEND_ID_PREFIX,
	BACKEND_PRESETS,
	BACKEND_PROTOCOLS,
	DEFAULT_BACKEND_CHAT_PATH,
	DEFAULT_BACKEND_MODELS_PATH,
	FAVICON_SERVICE_URL,
	LOCAL_BACKEND_ID
} from '$lib/constants';
import type {
	Backend,
	BackendCapabilities,
	BackendCompat,
	BackendPreset,
	BackendProtocol
} from '$lib/types';

/** Absolute chat completions URL for a backend. */
export function backendChatUrl(backend: Backend): string {
	return joinBackendUrl(backend.baseUrl, backend.chatPath ?? DEFAULT_BACKEND_CHAT_PATH);
}

/** Absolute models listing URL for a backend. */
export function backendModelsUrl(backend: Backend): string {
	return joinBackendUrl(backend.baseUrl, backend.modelsPath ?? DEFAULT_BACKEND_MODELS_PATH);
}

/** Icon size requested from the favicon service, shown at 16px. */
const FAVICON_SIZE = 64;

/**
 * Favicon of a backend's root domain, used when no bundled mark matches. API
 * hosts rarely serve a favicon themselves, so the subdomain is dropped
 * (`api.z.ai` -> `z.ai`) and the icon is requested through a favicon service.
 * Returns null when the URL carries no usable host.
 */
export function backendFaviconUrl(baseUrl: string): string | null {
	try {
		const labels = new URL(baseUrl).hostname.split('.');
		const root = labels.length > 2 ? labels.slice(-2).join('.') : labels.join('.');

		return root ? `${FAVICON_SERVICE_URL}${root}&sz=${FAVICON_SIZE}` : null;
	} catch {
		return null;
	}
}

/**
 * Preset a backend was created from, matched on origin and path so a saved
 * backend keeps its branding. Returns undefined for edited or custom URLs.
 */
export function findBackendPreset(baseUrl: string): BackendPreset | undefined {
	const target = normalizeBaseUrl(baseUrl);

	if (!target) return undefined;

	return BACKEND_PRESETS.find((preset) => normalizeBaseUrl(preset.baseUrl) === target);
}

/** Origin and path of a URL, without a trailing slash; null when unparsable. */
function normalizeBaseUrl(url: string): string | null {
	try {
		const parsed = new URL(url);

		return `${parsed.origin}${parsed.pathname.replace(/\/+$/, '')}`.toLowerCase();
	} catch {
		return null;
	}
}

/**
 * Features a backend supports, derived from its protocol. A missing backend
 * (unknown model, early startup) gets the plain compatible defaults.
 */
export function getBackendCapabilities(backend?: Backend): BackendCapabilities {
	return BACKEND_CAPABILITIES[backend?.protocol ?? 'openai'] ?? BACKEND_CAPABILITIES.openai;
}

/** Wire quirks for a backend: protocol defaults overridden by the backend. */
export function getBackendCompat(backend: Backend): BackendCompat {
	return { ...(BACKEND_COMPAT[backend.protocol] ?? BACKEND_COMPAT.openai), ...backend.compat };
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
		compat: parseBackendCompat(raw.compat, protocol),
		enabled: raw.enabled !== false,
		headers: parseBackendHeaders(raw.headers),
		id,
		modelsPath: parseOptionalPath(raw.modelsPath),
		name,
		protocol
	};
}

// only keep the override keys the protocol understands, so a stale persisted
// value can never inject an unknown field into a request
function parseBackendCompat(
	raw: unknown,
	protocol: BackendProtocol
): Partial<BackendCompat> | undefined {
	if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined;

	const entry = raw as Record<string, unknown>;
	const defaults = BACKEND_COMPAT[protocol] ?? BACKEND_COMPAT.openai;
	const overrides: Partial<BackendCompat> = {};

	if (entry.maxTokensField === 'max_tokens' || entry.maxTokensField === 'max_completion_tokens') {
		overrides.maxTokensField = entry.maxTokensField;
	}

	if (typeof entry.supportsUsageInStreaming === 'boolean') {
		overrides.supportsUsageInStreaming = entry.supportsUsageInStreaming;
	}

	// drop a no-op override so an unmodified backend stays undefined
	const isDefault =
		(overrides.maxTokensField === undefined ||
			overrides.maxTokensField === defaults.maxTokensField) &&
		(overrides.supportsUsageInStreaming === undefined ||
			overrides.supportsUsageInStreaming === defaults.supportsUsageInStreaming);

	return isDefault ? undefined : overrides;
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
