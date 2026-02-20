import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import TestReactiveMessagesWrapper from './components/TestReactiveMessagesWrapper.svelte';
import type { ChatRole, DatabaseMessage } from '$lib/types';

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

describe('ChatMessages reactivity to streaming additions', () => {
	it('updates reasoning block when new assistant tool-child arrives later', async () => {
		// reset store
		conversationsStore.activeMessages = [];

		const user = msg('u1', 'user', 'Question', null);
		const a1 = msg('a1', 'assistant', '', user.id, { thinking: 'step1' });
		const t1 = msg('t1', 'tool', JSON.stringify({ result: 'ok' }), a1.id, { toolCallId: 'call-1' });
		const a2 = msg('a2', 'assistant', '', t1.id, { thinking: 'step2' });

		// Render wrapper that consumes conversationsStore.activeMessages
		const { container } = render(TestReactiveMessagesWrapper);

		// Add initial chain (user + first assistant + tool)
		conversationsStore.addMessageToActive(user);
		conversationsStore.addMessageToActive(a1);
		conversationsStore.addMessageToActive(t1);

		// Initial reasoning shows step1 only
		await Promise.resolve();
		expect(container.textContent || '').toContain('step1');
		expect(container.textContent || '').not.toContain('step2');

		// Stream in follow-up assistant (same chain)
		conversationsStore.addMessageToActive(a2);

		// Wait a tick for UI to react
		await Promise.resolve();
		await Promise.resolve();

		const text = container.textContent || '';
		expect(text).toContain('step1');
		expect(text).toContain('step2');
	});
});
