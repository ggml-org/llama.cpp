import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

import { goto } from '$app/navigation';
import { CONVERSATION_TABS_LOCALSTORAGE_KEY, NEW_CHAT_TAB_ID } from '$lib/constants';
import { TabsStore } from '$lib/stores/tabs.svelte';

const gotoMock = vi.mocked(goto);

function freshStore(persisted?: unknown, validIds: string[] = []) {
	localStorage.removeItem(CONVERSATION_TABS_LOCALSTORAGE_KEY);

	if (persisted !== undefined) {
		localStorage.setItem(CONVERSATION_TABS_LOCALSTORAGE_KEY, JSON.stringify(persisted));
	}

	const tabs = new TabsStore();

	tabs.init(validIds);

	return tabs;
}

beforeEach(() => {
	gotoMock.mockClear();
});

/**
 * A plain navigation shows the new target in the current tab; only an
 * explicit new-tab action grows the strip.
 */
describe('tabsStore route sync', () => {
	it('creates the first tab, then navigates it in place', () => {
		const tabs = freshStore();

		tabs.syncWithRoute('a');
		tabs.syncWithRoute('b');
		tabs.syncWithRoute('c');

		expect(tabs.openTabs).toEqual(['c']);
	});

	it('activates an open tab instead of replacing the current one', () => {
		const tabs = freshStore(['a', 'b'], ['a', 'b']);

		tabs.syncWithRoute('b');
		tabs.syncWithRoute('a');

		expect(tabs.openTabs).toEqual(['a', 'b']);
	});

	it('turns the new-chat tab into the conversation it creates', async () => {
		const tabs = freshStore();

		tabs.syncWithRoute('a');
		await tabs.openInNewTab(NEW_CHAT_TAB_ID, { activate: true });
		tabs.syncWithRoute(NEW_CHAT_TAB_ID);
		tabs.syncWithRoute('b');

		expect(tabs.openTabs).toEqual(['a', 'b']);
	});

	it('waits for init before applying the route', () => {
		localStorage.setItem(CONVERSATION_TABS_LOCALSTORAGE_KEY, JSON.stringify(['a', 'b']));

		const tabs = new TabsStore();

		tabs.syncWithRoute('b');
		tabs.init(['a', 'b']);

		expect(tabs.openTabs).toEqual(['a', 'b']);
	});
});

describe('tabsStore explicit tabs', () => {
	it('opens new tabs at the end, in the background, without duplicates', async () => {
		const tabs = freshStore();

		tabs.syncWithRoute('a');
		await tabs.openInNewTab('b');
		await tabs.openInNewTab('c');
		await tabs.openInNewTab('b');

		expect(tabs.openTabs).toEqual(['a', 'b', 'c']);
		expect(gotoMock).not.toHaveBeenCalled();
	});

	it('closes the active tab to its left neighbour', async () => {
		const tabs = freshStore();

		tabs.syncWithRoute('a');
		await tabs.openInNewTab('b');
		tabs.syncWithRoute('b');
		await tabs.close('b', true);

		expect(tabs.openTabs).toEqual(['a']);
		expect(gotoMock).toHaveBeenLastCalledWith('#/chat/a');
	});

	it('falls back to the new-chat screen when the last tab closes', async () => {
		const tabs = freshStore();

		tabs.syncWithRoute('a');
		await tabs.close('a', true);

		expect(gotoMock).toHaveBeenLastCalledWith('#/');
	});
});

describe('tabsStore persistence', () => {
	it('prunes deleted conversations and keeps the new-chat tab', () => {
		const tabs = freshStore(['a', NEW_CHAT_TAB_ID, 'gone'], ['a']);

		expect(tabs.openTabs).toEqual(['a', NEW_CHAT_TAB_ID]);
	});
});
