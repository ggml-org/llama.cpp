import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { beforeEach, describe, expect, it } from 'vitest';

function mockProps(uiSettings: Record<string, string | number | boolean>) {
	Object.defineProperty(serverStore, 'props', {
		configurable: true,
		get: () =>
			({
				default_generation_settings: { params: { temperature: 0.8 } },
				ui_settings: uiSettings
			}) as unknown as typeof serverStore.props
	});
}

describe('server ui_settings application semantics', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
	});

	it('applies the admin defaults once for a new user', () => {
		settingsStore.initialize();
		mockProps({ apiKey: '', theme: 'dark' });

		settingsStore.syncWithServerDefaults();

		expect(settingsStore.config.theme).toBe('dark');
	});

	it('never reapplies on later loads: the user config diverges freely', () => {
		settingsStore.initialize();
		settingsStore.updateConfig('theme', 'light');
		settingsStore.updateConfig('apiKey', 'sk-user-key');

		// simulated F5: config now exists in localStorage
		settingsStore.initialize();
		mockProps({ apiKey: '', theme: 'dark' });

		settingsStore.syncWithServerDefaults();
		settingsStore.syncWithServerDefaults();

		expect(settingsStore.config.theme).toBe('light');
		expect(settingsStore.config.apiKey).toBe('sk-user-key');
		const stored = JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}');

		expect(stored.apiKey).toBe('sk-user-key');
	});

	it('Reset to Default reapplies the full baseline, api key included', () => {
		settingsStore.initialize();
		settingsStore.updateConfig('theme', 'light');
		settingsStore.updateConfig('apiKey', 'sk-user-key');
		mockProps({ apiKey: '', theme: 'dark' });

		settingsStore.forceSyncWithServerDefaults();

		expect(settingsStore.config.theme).toBe('dark');
		expect(settingsStore.config.apiKey).toBe('');
	});
});

describe('syncable scope (spec section 4)', () => {
	it('sampling params keep their live server twin, ui settings carry none', async () => {
		const { ParameterSyncService } = await import('$lib/services/parameter-sync.service');

		expect(ParameterSyncService.canSyncParameter('temperature')).toBe(true);
		expect(ParameterSyncService.canSyncParameter('samplers')).toBe(true);
		expect(ParameterSyncService.canSyncParameter('theme')).toBe(false);
		expect(ParameterSyncService.canSyncParameter('systemMessage')).toBe(false);
		expect(ParameterSyncService.canSyncParameter('apiKey')).toBe(false);
		expect(ParameterSyncService.canSyncParameter('customCss')).toBe(false);
	});
});
