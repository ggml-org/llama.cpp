import { DEFAULT_AGENTIC_CONFIG, SETTINGS_KEYS } from '$lib/constants';
import { describe, expect, it } from 'vitest';

/**
 * The registry `defaultValue` is the effective default: the store seeds its
 * config from SETTING_CONFIG_DEFAULT, so the loop's
 * `Number(setting) || DEFAULT_AGENTIC_CONFIG.maxTurns` fallback never fires.
 * Once these disagreed (10 vs 100) and the loop silently stopped early.
 */
describe('AGENTIC_MAX_TURNS default value', () => {
	it('sources the settings default from DEFAULT_AGENTIC_CONFIG.maxTurns', async () => {
		const { SETTING_CONFIG_DEFAULT } = await import('$lib/constants');

		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.AGENTIC_MAX_TURNS]).toBe(
			DEFAULT_AGENTIC_CONFIG.maxTurns
		);
	}, 15000);
});
