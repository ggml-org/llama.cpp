// Enable MCP is a real switch: while it is off, an added and enabled server is not usable.

import { SETTINGS_KEYS } from '#lib/constants/index.ts';
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import { mcpStore } from '#lib/stores/mcp/index.svelte.ts';
import { settingsStore } from '#lib/stores/settings/index.svelte.ts';
import { beforeEach, describe, expect, it } from 'vitest';

describe('Enable MCP setting', () => {
	beforeEach(() => {
		settingsStore.updateConfig(SETTINGS_KEYS.MCP_SERVERS, '[]');
		settingsStore.updateConfig(SETTINGS_KEYS.ENABLE_MCP, false);
		mcpStore.addServer({ enabled: true, url: 'https://mcp.example.com/a' });
	});

	it('keeps an enabled server out of use while it is off', () => {
		expect(mcpStore.isFeatureEnabled).toBe(false);
		expect(mcpStore.hasEnabledServers()).toBe(false);
		expect(mcpStore.hasAvailableServers()).toBe(false);
		expect(conversationsStore.preferences.policyEnabledServerIds()).toEqual([]);
	});

	it('makes the server usable once it is on', () => {
		settingsStore.updateConfig(SETTINGS_KEYS.ENABLE_MCP, true);

		expect(mcpStore.hasEnabledServers()).toBe(true);
		expect(mcpStore.hasAvailableServers()).toBe(true);
		expect(conversationsStore.preferences.policyEnabledServerIds()).toHaveLength(1);
	});

	it('keeps the server in the list either way, to manage it', () => {
		expect(mcpStore.getServers()).toHaveLength(1);
	});
});
