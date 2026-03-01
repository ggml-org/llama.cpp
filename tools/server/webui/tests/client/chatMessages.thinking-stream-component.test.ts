import { describe, expect, it } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import TestMessagesWrapper from './components/TestMessagesWrapper.svelte';
import type { DatabaseMessage } from '$lib/types';
import { conversationsStore } from '$lib/stores/conversations.svelte';

const makeMsg = (partial: Partial<DatabaseMessage>): DatabaseMessage => ({
	id: crypto.randomUUID(),
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

describe('ChatMessages component reasoning streaming', () => {
	it('keeps showing later reasoning chunks inline after a tool call', async () => {
		const user = makeMsg({ id: 'u1', role: 'user', type: 'text', content: 'hi' });
		const a1 = makeMsg({ id: 'a1', thinking: 'reasoning-step-1' });

		conversationsStore.activeMessages = [user, a1];

		const { container } = render(TestMessagesWrapper, {
			props: { messages: conversationsStore.activeMessages }
		});

		await waitFor(() => {
			expect(container.textContent || '').toContain('reasoning-step-1');
		});

		// Streamed tool call
		conversationsStore.updateMessageAtIndex(1, {
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: '{"expression":"1+1"}' }
				}
			])
		});

		// Tool result + next assistant continuation
		const tool = makeMsg({
			id: 't1',
			role: 'tool',
			type: 'tool',
			parent: 'a1',
			content: JSON.stringify({ result: '2' }),
			toolCallId: 'call-1'
		});
		const a2 = makeMsg({
			id: 'a2',
			parent: 't1',
			role: 'assistant',
			type: 'text',
			thinking: 'reasoning-step-2',
			content: 'final-answer'
		});

		conversationsStore.addMessageToActive(tool);
		conversationsStore.addMessageToActive(a2);

		await waitFor(() => {
			const text = container.textContent || '';
			expect(text).toContain('reasoning-step-1');
			expect(text).toContain('reasoning-step-2');
			expect(text).toContain('final-answer');
		});
	});
});
