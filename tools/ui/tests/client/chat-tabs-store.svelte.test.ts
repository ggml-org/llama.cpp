import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

import { goto } from '$app/navigation';
import {
	CONVERSATION_TABS_LOCALSTORAGE_KEY,
	NEW_CHAT_TAB_ID,
	PROJECT_HOME_TAB_ID
} from '$lib/constants';
import { TabsStore } from '$lib/stores/tabs.svelte';
import type { DatabaseConversation } from '$lib/types';
import { projectNewChatTabId, resolveTabView } from '$lib/utils/chat-tabs';

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

	it('opens a tab for each chat left from a project home, which has none', () => {
		const tabs = freshStore();

		tabs.syncWithRoute(PROJECT_HOME_TAB_ID);
		expect(tabs.openTabs).toEqual([]);

		tabs.syncWithRoute('x1');
		tabs.syncWithRoute(PROJECT_HOME_TAB_ID);
		tabs.syncWithRoute('x2');
		tabs.syncWithRoute('x3');

		expect(tabs.openTabs).toEqual(['x1', 'x3']);
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

/**
 * A project view (`#/projects/<id>/chat/...`) shows the chats of its project
 * and keeps them in it; the regular view shows every chat.
 */
describe('tab views', () => {
	const conversations = [
		{ id: 'g', name: 'g' },
		{ id: 'x', name: 'x', projectId: 'p1' },
		{ id: 'y', name: 'y', projectId: 'p1' }
	] as DatabaseConversation[];
	const open = ['g', 'x', projectNewChatTabId('p1'), 'y'];
	const projectChat = {
		params: { id: 'x', projectId: 'p1' },
		route: { id: '/(chat)/projects/[projectId]/chat/[id]' }
	};

	it('shows only the chats of the project, linked inside it', () => {
		const view = resolveTabView(projectChat, open, conversations);

		expect(view.ids).toEqual(['x', projectNewChatTabId('p1'), 'y']);
		expect(view.hrefFor('y')).toBe('#/projects/p1/chat/y');
		expect(view.hrefFor(projectNewChatTabId('p1'))).toBe('#/projects/p1/chat/new');
	});

	it('keeps the project tabs on the project home, none of them active', () => {
		const view = resolveTabView(
			{ params: { projectId: 'p1' }, route: { id: '/(chat)/projects/[projectId]' } },
			open,
			conversations
		);

		expect(view.activeId).toBeNull();
		expect(view.ids).toEqual(['x', projectNewChatTabId('p1'), 'y']);
		expect(view.hrefFor('x')).toBe('#/projects/p1/chat/x');
	});

	it("keeps a project's chats and new-chat tab out of the regular view", () => {
		const view = resolveTabView(
			{ params: { id: 'g' }, route: { id: '/(chat)/chat/[id]' } },
			open,
			conversations
		);

		expect(view.ids).toEqual(['g']);
		expect(view.hrefFor('g')).toBe('#/chat/g');
	});

	it('closes to a neighbour inside the project, then to the project page', async () => {
		const tabs = freshStore(open, ['g', 'x', 'y']);

		await tabs.close('y', true, resolveTabView(projectChat, tabs.openTabs, conversations));
		expect(gotoMock).toHaveBeenLastCalledWith('#/projects/p1/chat/new');

		await tabs.close('x', true, resolveTabView(projectChat, tabs.openTabs, conversations));
		await tabs.close(
			projectNewChatTabId('p1'),
			true,
			resolveTabView(projectChat, tabs.openTabs, conversations)
		);
		expect(gotoMock).toHaveBeenLastCalledWith('#/projects/p1');
		expect(tabs.openTabs).toEqual(['g']);
	});
});

describe('tabsStore persistence', () => {
	it('prunes deleted conversations and keeps the new-chat tabs', () => {
		const tabs = freshStore(['a', NEW_CHAT_TAB_ID, projectNewChatTabId('p1'), 'gone'], ['a']);

		expect(tabs.openTabs).toEqual(['a', NEW_CHAT_TAB_ID, projectNewChatTabId('p1')]);
	});

	it('merges tabs saved per project into one strip', () => {
		const tabs = freshStore(
			{
				global: { active: 'a', tabs: ['a'] },
				p1: { active: 'x', tabs: [PROJECT_HOME_TAB_ID, 'x'] }
			},
			['a', 'x']
		);

		expect(tabs.openTabs).toEqual(['a', 'x']);
	});
});
