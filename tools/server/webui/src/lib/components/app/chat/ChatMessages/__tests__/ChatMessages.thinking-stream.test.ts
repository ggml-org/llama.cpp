import { describe, expect, it } from 'vitest';
import { render, waitFor, cleanup } from '@testing-library/svelte';
import ChatMessages from '../ChatMessages.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import type { DatabaseMessage } from '$lib/types';

// deterministic IDs for test clarity
let idCounter = 0;
const uid = () => `m-${++idCounter}`;

const makeMsg = (partial: Partial<DatabaseMessage>): DatabaseMessage => ({
	id: uid(),
	convId: 'c1',
	role: 'assistant',
	type: 'text',
	parent: '-1',
	content: '',
	thinking: '',
	toolCalls: '',
	timestamp: Date.now(),
	children: [],
	...partial
});

describe('ChatMessages reasoning streaming', () => {
	it('renders consecutive reasoning chunks around a tool call inline without refresh', async () => {
		const user = makeMsg({ role: 'user', type: 'text', content: 'hi', id: 'u1' });
		const assistant1 = makeMsg({ id: 'a1', thinking: 'reasoning-step-1' });

		conversationsStore.activeMessages = [user, assistant1];

		const { container } = render(ChatMessages, {
			props: { messages: conversationsStore.activeMessages }
		});

		await waitFor(() => {
			expect(container.textContent || '').toContain('reasoning-step-1');
		});

		// Tool call arrives
		conversationsStore.updateMessageAtIndex(1, {
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: '{"expression":"1+1"}' }
				}
			])
		});

		// Tool message + continued assistant
		const tool = makeMsg({
			id: 't1',
			role: 'tool',
			type: 'tool',
			parent: 'a1',
			content: JSON.stringify({ result: '2' }),
			toolCallId: 'call-1'
		});
		const assistant2 = makeMsg({
			id: 'a2',
			parent: 't1',
			thinking: 'reasoning-step-2',
			content: 'final-answer'
		});
		conversationsStore.addMessageToActive(tool);
		conversationsStore.addMessageToActive(assistant2);

		await waitFor(() => {
			const text = container.textContent || '';
			expect(text).toContain('reasoning-step-1');
			expect(text).toContain('reasoning-step-2');
			expect(text).toContain('final-answer');
		});

		cleanup();
	});
});
