// Guards the legacy render-key migration: `renderUserContentAsMarkdown`
// and `renderThinkingAsMarkdown` (opt-INTO markdown) fold into the single
// `renderContentAsRawText` setting, with any explicit raw-text preference
// winning when the legacy keys disagree. Legacy keys are removed from the
// persisted config so they do not stay orphaned in localStorage.

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

describe('renderContentAsRawText migration', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
	});

	it('maps renderUserContentAsMarkdown=false to raw text', () => {
		seedConfig({ renderUserContentAsMarkdown: false });
		expect(config().renderContentAsRawText).toBe(true);
	});

	it('maps renderUserContentAsMarkdown=true to markdown', () => {
		seedConfig({ renderUserContentAsMarkdown: true });
		expect(config().renderContentAsRawText).toBe(false);
	});

	it('maps renderThinkingAsMarkdown=false to raw text', () => {
		seedConfig({ renderThinkingAsMarkdown: false });
		expect(config().renderContentAsRawText).toBe(true);
	});

	it('lets any explicit raw-text preference win when the legacy keys disagree', () => {
		seedConfig({ renderThinkingAsMarkdown: false, renderUserContentAsMarkdown: true });
		expect(config().renderContentAsRawText).toBe(true);
	});

	it('honors the intermediate renderUserContentAsRawText key from the PR branch', () => {
		seedConfig({ renderUserContentAsRawText: true });
		expect(config().renderContentAsRawText).toBe(true);
	});

	it('keeps an already-migrated value and cleans up the legacy keys', () => {
		seedConfig({ renderContentAsRawText: false, renderUserContentAsMarkdown: false });
		expect(config().renderContentAsRawText).toBe(false);

		const stored = persisted();

		expect(stored.renderUserContentAsMarkdown).toBeUndefined();
		expect(stored.renderThinkingAsMarkdown).toBeUndefined();
		expect(stored.renderUserContentAsRawText).toBeUndefined();
	});

	it('defaults to markdown when no legacy key exists', () => {
		seedConfig({});
		expect(config().renderContentAsRawText).toBe(false);
	});
});
