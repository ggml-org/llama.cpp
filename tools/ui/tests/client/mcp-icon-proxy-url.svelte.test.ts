// MCP icons live on a remote origin, which the browser cannot load under the
// UI's cross-origin isolation. When the user enabled the server's CORS proxy the
// icons are fetched through it; with the proxy off the origin URL stands in and
// the components fall back when it fails. Inline data: icons load under any
// policy and are never proxied.

import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { buildProxiedIconUrl } from '$lib/utils/cors-proxy';
import { beforeEach, describe, expect, it } from 'vitest';

// The favicon fallback resolves to the root domain of the server url, so the
// icons of https://mcp.example.com/mcp are fetched from https://example.com.
const FAVICON = 'https://example.com/favicon.ico';
const DATA_ICON = 'data:image/png;base64,iVBORw0KGgo=';

function mockProxyEnabled(enabled: boolean) {
	Object.defineProperty(serverStore, 'props', {
		configurable: true,
		get: () => ({ cors_proxy_enabled: enabled }) as unknown as typeof serverStore.props
	});
}

/** Target a proxied icon URL carries, or null when the URL is not proxied. */
function proxiedTarget(iconUrl: string | null): string | null {
	if (!iconUrl) return null;

	const url = new URL(iconUrl, window.location.origin);

	return url.pathname.endsWith('/cors-proxy') ? url.searchParams.get('url') : null;
}

describe('mcp server icon urls', () => {
	beforeEach(() => {
		settingsStore.updateConfig('mcpServers', '[]');
	});

	it('routes the favicon fallback through the proxy when it is enabled', () => {
		mockProxyEnabled(true);

		const server = mcpStore.addServer({ enabled: false, url: 'https://mcp.example.com/mcp' });

		expect(proxiedTarget(mcpStore.getServerFavicon(server.id))).toBe(FAVICON);
	});

	it('keeps the origin url when the proxy is off', () => {
		mockProxyEnabled(false);

		const server = mcpStore.addServer({ enabled: false, url: 'https://mcp.example.com/mcp' });

		expect(mcpStore.getServerFavicon(server.id)).toBe(FAVICON);
	});

	it('proxies an http icon url too', () => {
		expect(proxiedTarget(buildProxiedIconUrl('http://example.com/favicon.ico', true))).toBe(
			'http://example.com/favicon.ico'
		);
	});

	it('leaves a data url alone whichever way the proxy is set', () => {
		expect(buildProxiedIconUrl(DATA_ICON, true)).toBe(DATA_ICON);
		expect(buildProxiedIconUrl(DATA_ICON, false)).toBe(DATA_ICON);
		expect(buildProxiedIconUrl(null, true)).toBeNull();
	});
});
