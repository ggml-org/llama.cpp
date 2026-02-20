import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import TestSnapshotMessagesWrapper from './components/TestSnapshotMessagesWrapper.svelte';
import type { ChatRole, DatabaseMessage } from '$lib/types';

const waitForText = async (container: HTMLElement, text: string, timeoutMs = 2000) => {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if ((container.textContent || '').includes(text)) return;
		await new Promise((resolve) => setTimeout(resolve, 16));
	}
	throw new Error(`Timed out waiting for text: ${text}`);
};

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

/**
 * Reproduces the UI snapshot-passing pattern (messages={activeMessages()}).
 * Expects the final assistant content to appear after streaming additions.
 * This should fail until ChatScreen passes a reactive source.
 */
describe('ChatMessages snapshot regression', () => {
	it('fails to show final content when messages prop is a stale snapshot', async () => {
		conversationsStore.activeMessages = [];

		const user = msg('u1', 'user', 'Question', null);
		const a1 = msg('a1', 'assistant', '', user.id, {
			thinking: 'reasoning-step-1',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '1+1' }) }
				}
			])
		});
		const t1 = msg('t1', 'tool', JSON.stringify({ result: '2' }), a1.id, { toolCallId: 'call-1' });
		const a2 = msg('a2', 'assistant', '', t1.id, { thinking: 'reasoning-step-2' });
		const a3 = msg('a3', 'assistant', 'final-answer', a2.id);

		// Seed initial messages before render
		conversationsStore.addMessageToActive(user);
		conversationsStore.addMessageToActive(a1);
		conversationsStore.addMessageToActive(t1);

		const { container } = render(TestSnapshotMessagesWrapper);

		// Add later reasoning + final answer after mount (prop is stale snapshot)
		conversationsStore.addMessageToActive(a2);
		conversationsStore.addMessageToActive(a3);

		await waitForText(container, 'final-answer');

		// With a stale snapshot, final-answer will be missing; this expectation enforces correct behavior.
		expect(container.textContent || '').toContain('final-answer');
	});
});
