import { beforeEach, describe, expect, it } from 'vitest';
import { settingsStore, config } from '$lib/stores/settings.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { ParameterSyncService } from '$lib/services/parameter-sync.service';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-registry';
import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants/storage';
import type { SettingsConfigType } from '$lib/types';

type Primitive = string | number | boolean;

const KEYS = Object.keys(SETTING_CONFIG_DEFAULT).filter(
	(k) => ['string', 'number', 'boolean'].includes(typeof SETTING_CONFIG_DEFAULT[k]) && k !== 'theme'
);

function divergent(key: string, base: Primitive): Primitive {
	if (typeof base === 'boolean') return !base;
	if (typeof base === 'number') return base + 7;
	return `user-${key}`;
}

function baselineFor(key: string, base: Primitive): Primitive {
	if (typeof base === 'boolean') return !base;
	if (typeof base === 'number') return base + 42;
	return `admin-${key}`;
}

function mockProps(uiSettings: Record<string, Primitive>) {
	Object.defineProperty(serverStore, 'props', {
		configurable: true,
		get: () =>
			({
				default_generation_settings: { params: { temperature: 0.8 } },
				ui_settings: uiSettings
			}) as unknown as typeof serverStore.props
	});
}

const setUser = (key: string, value: Primitive) =>
	settingsStore.updateConfig(key as keyof SettingsConfigType, value as never);

const current = (key: string) => (config() as Record<string, unknown>)[key];

describe('registry-wide invariants', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
	});

	it('I1: no load ever modifies a stored user value, for any key of any type', () => {
		settingsStore.initialize();
		const userValues: Record<string, Primitive> = {};
		for (const key of KEYS) {
			userValues[key] = divergent(key, SETTING_CONFIG_DEFAULT[key] as Primitive);
			setUser(key, userValues[key]);
		}

		// simulated F5 + adverse admin baseline on every key, synced twice
		settingsStore.initialize();
		const adverse: Record<string, Primitive> = {};
		for (const key of KEYS)
			adverse[key] = baselineFor(key, SETTING_CONFIG_DEFAULT[key] as Primitive);
		mockProps(adverse);
		settingsStore.syncWithServerDefaults();
		settingsStore.syncWithServerDefaults();

		for (const key of KEYS) {
			expect(current(key), key).toBe(userValues[key]);
		}
	});

	it('first visit: the baseline applies for every key, false and 0 included', () => {
		settingsStore.initialize();
		const baseline: Record<string, Primitive> = {};
		for (const key of KEYS)
			baseline[key] = baselineFor(key, SETTING_CONFIG_DEFAULT[key] as Primitive);
		mockProps(baseline);

		settingsStore.syncWithServerDefaults();

		for (const key of KEYS) {
			if (ParameterSyncService.canSyncParameter(key)) continue;
			expect(current(key), key).toBe(baseline[key]);
		}
	});

	it('I3: Reset returns every key to baseline when defined, factory default otherwise', () => {
		settingsStore.initialize();
		for (const key of KEYS) setUser(key, divergent(key, SETTING_CONFIG_DEFAULT[key] as Primitive));

		const baseline: Record<string, Primitive> = {};
		KEYS.filter((_, i) => i % 2 === 0).forEach((key) => {
			baseline[key] = baselineFor(key, SETTING_CONFIG_DEFAULT[key] as Primitive);
		});
		mockProps(baseline);

		settingsStore.forceSyncWithServerDefaults();

		for (const key of KEYS) {
			if (key in baseline) {
				expect(current(key), key).toBe(baseline[key]);
			} else if (ParameterSyncService.canSyncParameter(key)) {
				expect(current(key), key).toBe('');
			} else {
				expect(current(key), key).toBe(SETTING_CONFIG_DEFAULT[key]);
			}
		}
	});
});
