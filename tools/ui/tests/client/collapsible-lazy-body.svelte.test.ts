// Guards the lazy-body behaviour of the shared collapsible wrappers.
//
// bits-ui's Collapsible.Content renders its children unconditionally and only
// sets `hidden`, so before this was gated a collapsed tool result kept its whole
// body in the DOM and re-rendered it on every streamed token (measured at
// ~41ms/section/token for a 200KB result). These tests pin the fix: closed means
// not rendered, and opening still mounts the body.

import CollapsibleLazyBodyHarness from './components/CollapsibleLazyBodyHarness.svelte';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const MARKER = 'collapsible-body-marker';

describe('collapsible wrappers render their body lazily', () => {
	for (const variant of ['content', 'terminal'] as const) {
		it(`${variant}: body is absent while closed and present once open`, async () => {
			const screen = render(CollapsibleLazyBodyHarness, { open: false, variant });

			await tick();

			expect(document.body.textContent).not.toContain(MARKER);

			await screen.rerender({ open: true, variant });
			await tick();

			expect(document.body.textContent).toContain(MARKER);

			// And it unmounts again on close, so a collapsed block stops costing
			// anything during streaming.
			await screen.rerender({ open: false, variant });
			await tick();

			expect(document.body.textContent).not.toContain(MARKER);
		});
	}
});
