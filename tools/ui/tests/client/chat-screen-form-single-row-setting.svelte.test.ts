// Guards the Display setting behind the single-row chat form: on by default, and off
// it leaves a one-line message on the two rows.

import ChatScreenFormTestWrapper from './components/ChatScreenFormTestWrapper.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { tick } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

async function isSingleRow() {
	const { container } = render(ChatScreenFormTestWrapper);

	await tick();

	return container.querySelector('[data-slot="input-area"]')!.hasAttribute('data-single-row');
}

describe('single-row chat form setting', () => {
	afterEach(() => {
		settingsStore.updateConfig(SETTINGS_KEYS.SINGLE_ROW_CHAT_FORM, true);
	});

	it('is on by default and can be turned off', async () => {
		await page.viewport(1280, 900);

		expect(await isSingleRow()).toBe(true);

		settingsStore.updateConfig(SETTINGS_KEYS.SINGLE_ROW_CHAT_FORM, false);
		expect(await isSingleRow()).toBe(false);
	});
});
