import { NEW_CHAT_TAB_ID } from '#lib/constants/index.ts';
import { resolveWindowCloseTab } from '#lib/utils/chat-tabs.ts';
import { describe, expect, it } from 'vitest';

const chatPage = (id: string) => ({ params: { id }, route: { id: '/(chat)/chat/[id]' } });
const newChatPage = { params: {}, route: { id: '/(chat)' } };
const settingsPage = { params: {}, route: { id: '/settings' } };

describe('resolveWindowCloseTab', () => {
	it('closes the active tab while other tabs stay open', () => {
		expect(resolveWindowCloseTab(chatPage('a'), ['a', 'b'])).toBe('a');
		expect(resolveWindowCloseTab(newChatPage, ['a', NEW_CHAT_TAB_ID])).toBe(NEW_CHAT_TAB_ID);
	});

	it('leaves the shortcut to the browser on the last tab', () => {
		expect(resolveWindowCloseTab(chatPage('a'), ['a'])).toBeNull();
	});

	it('leaves the shortcut to the browser on a page without tabs', () => {
		expect(resolveWindowCloseTab(settingsPage, ['a', 'b'])).toBeNull();
	});

	it('leaves the shortcut to the browser when the route has no tab', () => {
		expect(resolveWindowCloseTab(chatPage('c'), ['a', 'b'])).toBeNull();
	});
});
