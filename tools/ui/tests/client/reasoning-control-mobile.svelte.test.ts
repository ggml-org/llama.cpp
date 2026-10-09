// Guards the phone shape of the reasoning control: a phone picks the effort in a drawer,
// since a dropdown of small rows is not a touch target.

import ChatFormActionReasoning from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionReasoning.svelte';
import { ServerRole } from '$lib/enums';
import { serverStore } from '$lib/stores/server.svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };

describe('reasoning control on a phone', () => {
	beforeEach(async () => {
		serverStore.role = ServerRole.ROUTER;
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('offers the effort levels in a drawer', async () => {
		const screen = render(ChatFormActionReasoning);

		await screen.getByRole('button', { name: 'Reasoning effort' }).click();

		await expect.element(screen.getByRole('button', { name: /^Off$/ })).toBeVisible();
		await expect.element(screen.getByRole('button', { name: /^Max/ })).toBeVisible();
	});

	it('closes the drawer once a level is picked', async () => {
		const screen = render(ChatFormActionReasoning);

		await screen.getByRole('button', { name: 'Reasoning effort' }).click();
		await screen.getByRole('button', { name: 'Low' }).click();

		await expect.poll(() => screen.getByRole('button', { name: 'Low' }).query()).toBeNull();
	});
});
