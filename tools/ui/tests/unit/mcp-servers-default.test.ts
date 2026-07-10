import { describe, expect, it } from 'vitest';
import { SETTINGS_KEYS } from '$lib/constants/settings-keys';

/**
 * Default-value policy for the `MCP_SERVERS` setting.
 *
 * Earlier versions of the UI preloaded a hard-coded list of suggested
 * MCP servers into this setting on first install. That caused silent
 * third-party HTTP requests at app load (see issue #25509) and a popup
 * "recommendation" dialog (see issue #25274). New users must now opt
 * in explicitly when adding a server, so the default is an empty list.
 */
describe('MCP_SERVERS default value', () => {
	it('does not preload any servers in the MCP_SERVERS setting default', async () => {
		const { SETTING_CONFIG_DEFAULT } = await import('$lib/constants/settings-registry');

		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.MCP_SERVERS]).toBe('[]');
	}, 15000);
});
