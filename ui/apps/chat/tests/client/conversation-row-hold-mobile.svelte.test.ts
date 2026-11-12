// Guards the phone shape of a conversation row's actions: holding the row opens its menu
// anchored below it, while a tap still opens the chat and a drift is a scroll.

import ConversationRowHarness from './components/ConversationRowHarness.svelte';
import { ROW_LONG_PRESS_MS, ROW_LONG_PRESS_SLOP_PX } from '$lib/constants';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };
const conversation: DatabaseConversation = {
	currNode: null,
	id: 'conv-1',
	lastModified: Date.now(),
	name: 'Rust borrow checker help'
};

function touch(row: Element, type: string, dy = 0) {
	const rect = row.getBoundingClientRect();

	row.dispatchEvent(
		new PointerEvent(type, {
			bubbles: true,
			cancelable: true,
			clientX: rect.left + 20,
			clientY: rect.top + 10 + dy,
			isPrimary: true,
			pointerId: 1,
			pointerType: 'touch'
		})
	);
}

function wait(ms: number) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

describe('conversation row on a phone', () => {
	beforeEach(async () => {
		await page.viewport(PHONE.width, PHONE.height);
	});

	afterEach(async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('shows no three-dot trigger', async () => {
		const screen = render(ConversationRowHarness, { conversation });
		const row = screen.getByRole('button', { name: conversation.name });

		await row.hover();

		expect(screen.getByRole('button', { name: 'More actions' }).query()).toBeNull();
	});

	it('opens the actions menu on a hold, without opening the chat', async () => {
		const onSelect = vi.fn();
		const screen = render(ConversationRowHarness, { conversation, onSelect });
		const row = screen.getByRole('button', { name: conversation.name }).element();

		touch(row, 'pointerdown');
		await wait(ROW_LONG_PRESS_MS + 50);
		touch(row, 'pointerup');
		(row as HTMLElement).click();

		await expect.element(page.getByRole('menuitem', { name: 'Delete' })).toBeVisible();
		expect(onSelect).not.toHaveBeenCalled();
	});

	it('treats a drift as a scroll, not a hold', async () => {
		const screen = render(ConversationRowHarness, { conversation });
		const row = screen.getByRole('button', { name: conversation.name }).element();

		touch(row, 'pointerdown');
		touch(row, 'pointermove', ROW_LONG_PRESS_SLOP_PX * 2);
		await wait(ROW_LONG_PRESS_MS + 50);

		expect(page.getByRole('menuitem').query()).toBeNull();
	});

	it('opens the chat on a tap', async () => {
		const onSelect = vi.fn();
		const screen = render(ConversationRowHarness, { conversation, onSelect });
		const row = screen.getByRole('button', { name: conversation.name }).element();

		touch(row, 'pointerdown');
		touch(row, 'pointerup');
		(row as HTMLElement).click();

		expect(onSelect).toHaveBeenCalledWith(conversation.id);
		expect(page.getByRole('menuitem').query()).toBeNull();
	});
});
