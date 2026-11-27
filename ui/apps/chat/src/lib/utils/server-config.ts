import { withBase } from './base-path';

/**
 * Where the llama-server is. By default it serves the app, so its API is on the app's own
 * origin. An app hosted apart from it (a static host, `npx llama-chat --server`) names it in
 * `config.json`, a file next to index.html: `{ "server": "http://localhost:9931" }`.
 */

/** The file next to index.html that holds the settings a host gives the app. */
export const SERVER_CONFIG_FILE = '/config.json';

let serverBase = '';

/**
 * The server's address as a prefix for an API path: its origin and any path it is served
 * under, without a trailing slash. Empty for anything but an http(s) URL.
 */
export function normalizeServerBase(value: unknown): string {
	if (typeof value !== 'string' || !value.trim()) return '';

	try {
		const url = new URL(value.trim());

		if (url.protocol !== 'http:' && url.protocol !== 'https:') return '';

		return `${url.origin}${url.pathname.replace(/\/+$/, '')}`;
	} catch {
		return '';
	}
}

/** Names the server, as `config.json` does. Empty puts it back on the app's own origin. */
export function setServerBase(value: unknown): void {
	serverBase = normalizeServerBase(value);
}

/** The server named by `config.json`, or an empty string when it is on the app's own origin. */
export function getServerBase(): string {
	return serverBase;
}

/**
 * Reads `config.json` once, before the app starts. A missing or broken file leaves the server
 * on the app's own origin, as does a host that is offline.
 */
export async function loadServerConfig(): Promise<void> {
	try {
		const response = await fetch(withBase(SERVER_CONFIG_FILE), { cache: 'no-store' });

		if (!response.ok) return;

		const config: unknown = await response.json();

		if (config && typeof config === 'object' && 'server' in config) setServerBase(config.server);
	} catch {
		// no config.json: the server is the one that serves the app
	}
}

/** The URL of a server path (`/props`, `./props`): on the configured server, else on the app's origin. */
export function serverUrl(path: string): string {
	return serverBase ? `${serverBase}/${path.replace(/^\.?\//, '')}` : withBase(path);
}
