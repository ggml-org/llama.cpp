// Regression test: opening a chat-form picker must not scroll the
// conversation to the top. Root cause: the list's scroll effect fired
// scrollIntoView on the initial mount, before the popover was positioned,
// so the browser scrolled every scrollable ancestor to reveal the row.

import PickerListScrollHarness from './components/PickerListScrollHarness.svelte';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

describe('ChatFormPickerList mount scroll', () => {
	it('does not scroll documentElement when the picker mounts', async () => {
		const screen = render(PickerListScrollHarness);

		await tick();

		document.documentElement.scrollTop = document.documentElement.scrollHeight;
		await tick();
		const before = document.documentElement.scrollTop;

		expect(before).toBeGreaterThan(0);

		screen.component.openPicker();
		await tick();
		await new Promise((r) => setTimeout(r, 100));
		await tick();

		const after = document.documentElement.scrollTop;

		expect(after).toBe(before);
	});
});
