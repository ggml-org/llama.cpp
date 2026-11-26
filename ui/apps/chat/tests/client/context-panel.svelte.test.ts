// Guards how the context panel opens: a phone shows it in a drawer, while a wider screen opens a
// popover from the gauge on click (not on hover) and closes it on Escape or a click outside.

import ChatFormContextGauge from '#lib/components/app/chat/ChatForm/ChatFormContextGauge/ChatFormContextGauge.svelte';
import ContextGaugeDrawer from '#lib/components/app/chat/ChatForm/ChatFormContextGauge/ContextGaugeDrawer.svelte';
import { gaugePopup } from '#lib/components/app/chat/ChatForm/ChatFormContextGauge/gauge-popup.svelte.ts';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };

describe('context panel', () => {
	beforeEach(async () => {
		gaugePopup.open = false;
	});

	afterEach(async () => {
		gaugePopup.open = false;
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('shows the panel in a drawer on a phone', async () => {
		await page.viewport(PHONE.width, PHONE.height);

		const screen = render(ContextGaugeDrawer);

		gaugePopup.open = true;

		await expect.element(screen.getByRole('heading', { name: 'Context' })).toBeVisible();
	});

	it('opens the popover on click, not on hover, on a wider screen', async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);

		render(ChatFormContextGauge);

		const trigger = page.getByRole('button', { name: 'Context usage' });

		await trigger.hover();
		expect(gaugePopup.open).toBe(false);

		await trigger.click();
		await expect.element(page.getByRole('dialog')).toBeVisible();

		await userEvent.keyboard('{Escape}');
		await expect.poll(() => gaugePopup.open).toBe(false);
	});

	it('closes the popover on a click outside', async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);

		render(ChatFormContextGauge);

		await page.getByRole('button', { name: 'Context usage' }).click();
		await expect.element(page.getByRole('dialog')).toBeVisible();

		await userEvent.click(document.body, { position: { x: 5, y: 5 } });
		await expect.poll(() => gaugePopup.open).toBe(false);
	});
});
