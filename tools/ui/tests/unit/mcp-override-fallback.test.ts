import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants';
import { SETTINGS_KEYS } from '$lib/constants/settings-keys';
import type { DatabaseConversation } from '$lib/types/database';

// node env unit project has no DOM, install a minimal localStorage backed by a Map
beforeAll(() => {
	const store = new Map<string, string>();
	const polyfill: Storage = {
		get length() {
			return store.size;
		},
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};
	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;
});

/**
 * Regression coverage for the bug where MCP servers flipped to "disabled"
 * after sending the first message on a fresh chat (see comment in
 * `MCPStore.createConversation`: empty `mcpServerOverrides` should inherit
 * `mcpServers[i].enabled`, not be treated as all-off).
 */
describe('conversationsStore MCP override resolution', () => {
	beforeEach(async () => {
		localStorage.clear();
		// Two configured servers: alpha is globally disabled, bravo enabled.
		localStorage.setItem(
			CONFIG_LOCALSTORAGE_KEY,
			JSON.stringify({
				[SETTINGS_KEYS.MCP_SERVERS]: JSON.stringify([
					{ id: 'alpha', enabled: false, url: 'https://alpha.example.com/mcp' },
					{ id: 'bravo', enabled: true, url: 'https://bravo.example.com/mcp' }
				])
			})
		);

		// The settings store constructor bails in node env (no `browser`),
		// so seed the config directly. The shape mirrors what `loadConfig`
		// would build from localStorage.
		const { settingsStore } = await import('$lib/stores/settings.svelte');
		const raw = localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}';
		const saved = JSON.parse(raw) as Record<string, unknown>;
		settingsStore.config = {
			...settingsStore.config,
			[SETTINGS_KEYS.MCP_SERVERS]: saved[SETTINGS_KEYS.MCP_SERVERS]
		};
	});

	afterEach(() => {
		localStorage.clear();
	});

	function makeConversation(
		overrides?: { serverId: string; enabled: boolean }[]
	): DatabaseConversation {
		return {
			id: 'conv-1',
			currNode: null,
			lastModified: 0,
			name: 'Test chat',
			mcpServerOverrides: overrides
		};
	}

	it('inherits server.enabled when no conversation is active', async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		conversationsStore.activeConversation = null;

		expect(conversationsStore.isMcpServerEnabledForChat('alpha')).toBe(false);
		expect(conversationsStore.isMcpServerEnabledForChat('bravo')).toBe(true);
	});

	it('inherits server.enabled on a newly created chat with no overrides', async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		conversationsStore.activeConversation = makeConversation();

		// Empty override list: must fall back to global server.enabled, not all-off.
		expect(conversationsStore.isMcpServerEnabledForChat('alpha')).toBe(false);
		expect(conversationsStore.isMcpServerEnabledForChat('bravo')).toBe(true);
	});

	it('inherits server.enabled on a newly created chat when overrides is undefined', async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		conversationsStore.activeConversation = makeConversation(undefined);

		expect(conversationsStore.isMcpServerEnabledForChat('alpha')).toBe(false);
		expect(conversationsStore.isMcpServerEnabledForChat('bravo')).toBe(true);
	});

	it('uses explicit per-chat overrides, with defaults for non-overridden servers', async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		// Override flips bravo off for this chat, alpha keeps its global default.
		conversationsStore.activeConversation = makeConversation([
			{ serverId: 'bravo', enabled: false }
		]);

		expect(conversationsStore.isMcpServerEnabledForChat('alpha')).toBe(false);
		expect(conversationsStore.isMcpServerEnabledForChat('bravo')).toBe(false);
	});

	it('getAllMcpServerOverrides returns a complete list merged from defaults', async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		conversationsStore.activeConversation = makeConversation([
			{ serverId: 'alpha', enabled: true }
		]);

		expect(conversationsStore.getAllMcpServerOverrides()).toEqual([
			{ serverId: 'alpha', enabled: true },
			{ serverId: 'bravo', enabled: true }
		]);
	});

	it('getAllMcpServerOverrides falls back to defaults when there are no explicit overrides', async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		conversationsStore.activeConversation = makeConversation();

		expect(conversationsStore.getAllMcpServerOverrides()).toEqual([
			{ serverId: 'alpha', enabled: false },
			{ serverId: 'bravo', enabled: true }
		]);
	});

	it('getMcpServerOverride returns the global default when the server has no explicit override', async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		conversationsStore.activeConversation = makeConversation([
			{ serverId: 'alpha', enabled: true }
		]);

		expect(conversationsStore.getMcpServerOverride('bravo')).toEqual({
			serverId: 'bravo',
			enabled: true
		});
	});
});
