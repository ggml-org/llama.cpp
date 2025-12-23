import { describe, it, expect, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { tick } from 'svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import type { ChatRole, DatabaseMessage } from '$lib/types';
import TestChatMessageWrapper from './components/TestChatMessageWrapper.svelte';

const msg = (
	id: string,
	role: ChatRole,
	content: string,
	parent: string | null,
	extra: Partial<DatabaseMessage> = {}
): DatabaseMessage => ({
	id,
	convId: 'c1',
	type: 'text',
	role,
	content,
	thinking: '',
	toolCalls: '',
	parent: parent ?? '-1',
	children: [],
	timestamp: Date.now(),
	...extra
});

async function waitForDialogContent(): Promise<HTMLElement> {
	for (let i = 0; i < 20; i++) {
		await tick();
		const dialog = document.body.querySelector(
			'[data-slot="alert-dialog-content"]'
		) as HTMLElement | null;
		if (dialog) return dialog;
		await new Promise((r) => setTimeout(r, 0));
	}
	throw new Error('Timed out waiting for delete confirmation dialog');
}

describe('ChatMessage delete for merged assistant messages', () => {
	it('deletes the actionTarget message id (not the merged display id)', async () => {
		settingsStore.config = { ...SETTING_CONFIG_DEFAULT };

		conversationsStore.activeConversation = {
			id: 'c1',
			name: 'Test',
			currNode: null,
			lastModified: Date.now()
		};

		// Chain: user -> assistant(toolcall) -> tool -> assistant(final)
		const user = msg('u1', 'user', 'Question', null, { children: ['a1'] });
		const a1 = msg('a1', 'assistant', '', user.id, {
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '1+1' }) }
				}
			]),
			children: ['t1']
		});
		const t1 = msg('t1', 'tool', JSON.stringify({ result: '2' }), a1.id, {
			toolCallId: 'call-1',
			children: ['a2']
		});
		const a2 = msg('a2', 'assistant', 'Answer is 2', t1.id);
		const allMessages = [user, a1, t1, a2];

		// Merged display message: looks like a2, but actions should target a1.
		const mergedAssistant = { ...a2, _actionTargetId: a1.id } as unknown as DatabaseMessage;

		conversationsStore.activeMessages = allMessages;

		// Avoid touching IndexedDB by stubbing the store call used by getDeletionInfo.
		const originalGetConversationMessages =
			conversationsStore.getConversationMessages.bind(conversationsStore);
		conversationsStore.getConversationMessages = async () => allMessages;

		const onDelete = vi.fn();

		try {
			const { container } = render(TestChatMessageWrapper, {
				target: document.body,
				props: { message: mergedAssistant, onDelete }
			});

			const deleteButton = container.querySelector(
				'button[aria-label="Delete"]'
			) as HTMLButtonElement | null;
			expect(deleteButton).toBeTruthy();

			deleteButton?.click();

			const dialog = await waitForDialogContent();
			const confirm = Array.from(dialog.querySelectorAll('button')).find((b) =>
				(b.textContent ?? '').includes('Delete')
			) as HTMLButtonElement | undefined;
			expect(confirm).toBeTruthy();

			confirm?.click();
			await tick();

			expect(onDelete).toHaveBeenCalledTimes(1);
			expect(onDelete.mock.calls[0][0].id).toBe(a1.id);
		} finally {
			conversationsStore.getConversationMessages = originalGetConversationMessages;
		}
	});
});
