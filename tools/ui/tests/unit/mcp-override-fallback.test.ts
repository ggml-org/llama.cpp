import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { CONFIG_LOCALSTORAGE_KEY, SETTINGS_KEYS } from '$lib/constants';
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
 * Regression coverage for MCP servers flipping to disabled after the first
 * message on a fresh chat. Global server availability and per-conversation
 * tool policy are separate inputs; an empty conversation policy must not
 * disable every globally enabled server.
 */
describe('conversation MCP tool policy resolution', () => {
	beforeEach(async () => {
		localStorage.clear();
		localStorage.setItem(
			CONFIG_LOCALSTORAGE_KEY,
			JSON.stringify({
				[SETTINGS_KEYS.MCP_SERVERS]: JSON.stringify([
					{ id: 'alpha', enabled: false, url: 'https://alpha.example.com/mcp' },
					{ id: 'bravo', enabled: true, url: 'https://bravo.example.com/mcp' }
				])
			})
		);

		// Stores must load after the test installs localStorage; a static import
		// would initialize their singleton state before the polyfill exists.
		const { settingsStore } = await import('$lib/stores');
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

	function makeConversation(disabledTools?: string[]): DatabaseConversation {
		return {
			id: 'conv-1',
			currNode: null,
			disabledTools,
			lastModified: 0,
			name: 'Test chat'
		};
	}

	it('uses global server availability when no conversation is active', async () => {
		const { conversationsStore } = await import('$lib/stores');
		conversationsStore.activeConversation = null;

		expect(conversationsStore.preferences.policyEnabledServerIds()).toEqual(['bravo']);
	});

	it('preserves globally enabled servers for a new chat with no policy', async () => {
		const { conversationsStore } = await import('$lib/stores');
		conversationsStore.activeConversation = makeConversation();

		expect(conversationsStore.preferences.policyEnabledServerIds()).toEqual(['bravo']);
	});

	it('applies an explicit per-chat MCP server disable', async () => {
		const { conversationsStore, toolsStore } = await import('$lib/stores');
		conversationsStore.activeConversation = makeConversation([
			toolsStore.getMcpServerToolsKey('bravo')
		]);

		expect(conversationsStore.preferences.policyEnabledServerIds()).toEqual([]);
	});
});
