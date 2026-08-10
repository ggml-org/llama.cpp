// Guards the unfolding of `renderContentAsRawText` back onto the two
// per-surface render keys. The single toggle carried user content and
// thinking at once, so only the user key is restored from it and thinking
// returns to its own default. The toggle is removed from the persisted
// config so it does not stay orphaned in localStorage.

import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants/storage';
import { config, settingsStore } from '$lib/stores/settings.svelte';
import { beforeEach, describe, expect, it } from 'vitest';

function seedConfig(stored: Record<string, unknown>) {
	localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(stored));
	settingsStore.initialize();
}

function persisted(): Record<string, unknown> {
	return JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}');
}

describe('renderContentAsRawText unfolding', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
	});

	it('maps raw text to user content as plain text', () => {
		seedConfig({ renderContentAsRawText: true });
		expect(config().renderUserContentAsMarkdown).toBe(false);
	});

	it('maps markdown to user content as markdown', () => {
		seedConfig({ renderContentAsRawText: false });
		expect(config().renderUserContentAsMarkdown).toBe(true);
	});

	it('leaves thinking on its own default', () => {
		seedConfig({ renderContentAsRawText: true });
		expect(config().renderThinkingAsMarkdown).toBe(true);
	});

	it('keeps an explicit user preference over the toggle', () => {
		seedConfig({ renderContentAsRawText: true, renderUserContentAsMarkdown: true });
		expect(config().renderUserContentAsMarkdown).toBe(true);
	});

	it('drops the toggle from the persisted config', () => {
		seedConfig({ renderContentAsRawText: true });
		expect(persisted().renderContentAsRawText).toBeUndefined();
	});

	it('leaves both surfaces on markdown when nothing is stored', () => {
		seedConfig({});
		expect(config().renderUserContentAsMarkdown).toBe(true);
		expect(config().renderThinkingAsMarkdown).toBe(true);
	});
});
