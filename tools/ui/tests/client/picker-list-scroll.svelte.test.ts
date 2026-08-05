// Regression test: opening a chat-form picker (mention `@` / command `/`) must
// not scroll the conversation (documentElement) to the top.
//
// Root cause: ChatFormPickerList's scroll effect fired scrollIntoView on the
// initial mount (lastScrollTrigger started at null, scrollTrigger at 0), before
// the popover was positioned, so the browser scrolled every scrollable ancestor
// - including documentElement - to reveal the first row.

import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { tick } from 'svelte';
import PickerListScrollHarness from './components/PickerListScrollHarness.svelte';

describe('ChatFormPickerList mount scroll', () => {
	it('does not scroll documentElement when the picker mounts', async () => {
		const screen = render(PickerListScrollHarness);
		await tick();

		// Scroll to the bottom first, like a long conversation.
		document.documentElement.scrollTop = document.documentElement.scrollHeight;
		await tick();
		const before = document.documentElement.scrollTop;
		expect(before).toBeGreaterThan(0);

		// Mount the picker list (this is what happens when the popover opens).
		screen.component.openPicker();
		await tick();
		await new Promise((r) => setTimeout(r, 100));
		await tick();

		const after = document.documentElement.scrollTop;
		expect(after).toBe(before);
	});
});
