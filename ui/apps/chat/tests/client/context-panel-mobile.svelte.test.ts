// Guards the phone shape of the context panel: a phone shows it in a drawer, while a
// wider screen keeps the card anchored above the dial.

import ContextGaugePopup from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/ContextGaugePopup.svelte';
import { gaugePopup } from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/gauge-popup.svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };

describe('context panel on a phone', () => {
	beforeEach(async () => {
		gaugePopup.open = false;
	});

	afterEach(async () => {
		gaugePopup.open = false;
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('shows the panel in a drawer', async () => {
		await page.viewport(PHONE.width, PHONE.height);

		const screen = render(ContextGaugePopup);

		gaugePopup.open = true;

		await expect.element(screen.getByRole('heading', { name: 'Context' })).toBeVisible();
	});

	it('keeps the anchored card on a wider screen', async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);

		const screen = render(ContextGaugePopup);

		gaugePopup.open = true;

		await expect.element(screen.getByRole('status')).toBeVisible();
	});
});
