// Guards the unfolding of `renderContentAsRawText` back onto the two
// per-surface render keys. The single toggle carried user content and
// thinking at once, so only the user key is restored from it and thinking
// returns to its own default. The toggle is removed from the persisted
// config so it does not stay orphaned in localStorage.

import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants';
import { MigrationService } from '$lib/services/migration.service';
import { settingsStore } from '$lib/stores/settings.svelte';
import { beforeEach, describe, expect, it } from 'vitest';

const RENDER_KEYS_MIGRATION_ID = 'render-keys-unfold-v1';

async function seedConfig(stored: Record<string, unknown>) {
	localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(stored));

	const migration = MigrationService.getMigrations().find((m) => m.id === RENDER_KEYS_MIGRATION_ID);

	await migration?.run();
	settingsStore.initialize();
}

function persisted(): Record<string, unknown> {
	return JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}');
}

describe('renderContentAsRawText unfolding', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		MigrationService.resetState();
		settingsStore.initialize();
	});

	it('maps raw text to user content as plain text', async () => {
		await seedConfig({ renderContentAsRawText: true });
		expect(settingsStore.config.renderUserContentAsMarkdown).toBe(false);
	});

	it('maps markdown to user content as markdown', async () => {
		await seedConfig({ renderContentAsRawText: false });
		expect(settingsStore.config.renderUserContentAsMarkdown).toBe(true);
	});

	it('leaves thinking on its own default', async () => {
		await seedConfig({ renderContentAsRawText: true });
		expect(settingsStore.config.renderThinkingAsMarkdown).toBe(true);
	});

	it('keeps an explicit user preference over the toggle', async () => {
		await seedConfig({ renderContentAsRawText: true, renderUserContentAsMarkdown: true });
		expect(settingsStore.config.renderUserContentAsMarkdown).toBe(true);
	});

	it('drops the toggle from the persisted config', async () => {
		await seedConfig({ renderContentAsRawText: true });
		expect(persisted().renderContentAsRawText).toBeUndefined();
	});

	it('leaves both surfaces on markdown when nothing is stored', async () => {
		await seedConfig({});
		expect(settingsStore.config.renderUserContentAsMarkdown).toBe(true);
		expect(settingsStore.config.renderThinkingAsMarkdown).toBe(true);
	});
});
