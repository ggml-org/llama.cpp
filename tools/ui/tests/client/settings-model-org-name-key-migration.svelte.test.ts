// Guards the rename of `showModelOrgNameInTrigger` onto `showModelOrgName`.
// The persisted config always holds the legacy key, so a value it carries moves
// over, while a config that already set the new key keeps its own value. The
// legacy key is removed so it does not stay orphaned in localStorage.

import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants';
import { MigrationService } from '$lib/services/migration.service';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { beforeEach, describe, expect, it } from 'vitest';

const MODEL_ORG_NAME_KEY_MIGRATION_ID = 'model-org-name-key-v1';

async function seedConfig(stored: Record<string, unknown>) {
	localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(stored));

	const migration = MigrationService.getMigrations().find(
		(m) => m.id === MODEL_ORG_NAME_KEY_MIGRATION_ID
	);

	await migration?.run();
	settingsStore.initialize();
}

function persisted(): Record<string, unknown> {
	return JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}');
}

describe('showModelOrgNameInTrigger rename', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		MigrationService.resetState();
		settingsStore.initialize();
	});

	it('carries an enabled trigger over to the selector-wide key', async () => {
		await seedConfig({ showModelOrgNameInTrigger: true });
		expect(settingsStore.config.showModelOrgName).toBe(true);
	});

	it('carries a disabled trigger over to the selector-wide key', async () => {
		await seedConfig({ showModelOrgNameInTrigger: false });
		expect(settingsStore.config.showModelOrgName).toBe(false);
	});

	it('keeps an explicit user preference over the legacy key', async () => {
		await seedConfig({ showModelOrgName: false, showModelOrgNameInTrigger: true });
		expect(settingsStore.config.showModelOrgName).toBe(false);
	});

	it('drops the legacy key from the persisted config', async () => {
		await seedConfig({ showModelOrgNameInTrigger: true });
		expect(persisted().showModelOrgNameInTrigger).toBeUndefined();
	});

	it('leaves the selector-wide key on its default when nothing is stored', async () => {
		await seedConfig({});
		expect(settingsStore.config.showModelOrgName).toBe(true);
	});
});
