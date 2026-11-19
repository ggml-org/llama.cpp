// Guards that opening a panel on a phone leaves its corner close alone: the tap that
// opened the panel is already inside it, so a focus ring on that close reads as a stray
// highlight.

import DialogPanelWrapper from './components/DialogPanelWrapper.svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };

describe.each(['settings', 'mcp'] as const)('%s panel on a phone', (panel) => {
	beforeEach(async () => {
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('leaves the corner close unfocused', async () => {
		const screen = render(DialogPanelWrapper, { panel });

		await expect.element(screen.getByRole('dialog')).toBeVisible();

		// the panel focuses its first button on open unless it is told not to
		await expect.poll(() => document.activeElement?.tagName).not.toBe('BUTTON');
	});
});
