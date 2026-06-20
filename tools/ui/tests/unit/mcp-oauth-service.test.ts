import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BrowserMcpOAuthProvider } from '$lib/services/mcp-oauth.service';
import { MCP_OAUTH_LOCALSTORAGE_KEY, MCP_OAUTH_STATUS_LOCALSTORAGE_KEY } from '$lib/constants';

class MemoryStorage implements Storage {
	private values = new Map<string, string>();

	get length(): number {
		return this.values.size;
	}

	clear(): void {
		this.values.clear();
	}

	getItem(key: string): string | null {
		return this.values.get(key) ?? null;
	}

	key(index: number): string | null {
		return Array.from(this.values.keys())[index] ?? null;
	}

	removeItem(key: string): void {
		this.values.delete(key);
	}

	setItem(key: string, value: string): void {
		this.values.set(key, value);
	}
}

describe('BrowserMcpOAuthProvider', () => {
	beforeEach(() => {
		vi.stubGlobal('localStorage', new MemoryStorage());
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('persists client information and tokens by MCP server id', () => {
		const provider = new BrowserMcpOAuthProvider({
			serverId: 'oauth-server',
			serverUrl: 'https://mcp.example.com/mcp',
			redirectUrl: 'https://chat.example/mcp-callback',
			returnUrl: 'https://chat.example/settings'
		});

		provider.saveClientInformation({ client_id: 'client-123' });
		provider.saveTokens({ access_token: 'token-123', token_type: 'Bearer' });

		const restored = new BrowserMcpOAuthProvider({
			serverId: 'oauth-server',
			serverUrl: 'https://mcp.example.com/mcp',
			redirectUrl: 'https://chat.example/mcp-callback',
			returnUrl: 'https://chat.example/settings'
		});

		expect(restored.clientInformation()).toEqual({ client_id: 'client-123' });
		expect(restored.tokens()).toEqual({ access_token: 'token-123', token_type: 'Bearer' });
		expect(localStorage.getItem(MCP_OAUTH_LOCALSTORAGE_KEY)).toContain('oauth-server');
	});

	it('validates protected resource metadata against the original MCP server URL', async () => {
		const provider = new BrowserMcpOAuthProvider({
			serverId: 'oauth-server',
			serverUrl: 'https://mcp.example.com/mcp',
			redirectUrl: 'https://chat.example/mcp-callback',
			returnUrl: 'https://chat.example/settings'
		});

		await expect(
			provider.validateResourceURL(
				'https://chat.example/cors-proxy?url=https%3A%2F%2Fmcp.example.com%2Fmcp',
				'https://mcp.example.com/mcp'
			)
		).resolves.toEqual(new URL('https://mcp.example.com/mcp'));

		await expect(
			provider.validateResourceURL(
				'https://chat.example/cors-proxy?url=https%3A%2F%2Fmcp.example.com%2Fmcp',
				'https://other.example.com/mcp'
			)
		).rejects.toThrow('does not match MCP server');
	});

	it('publishes authorization URL without opening a browser tab automatically', () => {
		const open = vi.fn();
		vi.stubGlobal('open', open);
		const provider = new BrowserMcpOAuthProvider({
			serverId: 'oauth-server',
			serverUrl: 'https://mcp.example.com/mcp',
			redirectUrl: 'https://chat.example/mcp-callback',
			returnUrl: 'https://chat.example/settings'
		});

		provider.redirectToAuthorization(new URL('https://auth.example.com/oauth/authorize'));

		expect(open).not.toHaveBeenCalled();
		expect(localStorage.getItem(MCP_OAUTH_STATUS_LOCALSTORAGE_KEY)).toContain(
			'https://auth.example.com/oauth/authorize'
		);
	});

	it('navigates a user-opened authorization window when one is registered', () => {
		const authorizationWindow = {
			closed: false,
			location: { href: 'about:blank' }
		} as Window;
		const provider = new BrowserMcpOAuthProvider({
			serverId: 'oauth-server',
			serverUrl: 'https://mcp.example.com/mcp',
			redirectUrl: 'https://chat.example/mcp-callback',
			returnUrl: 'https://chat.example/settings'
		});

		BrowserMcpOAuthProvider.beginInteractiveAuthorization(authorizationWindow);
		provider.redirectToAuthorization(new URL('https://auth.example.com/oauth/authorize'));

		expect(authorizationWindow.location.href).toBe('https://auth.example.com/oauth/authorize');
	});
});
