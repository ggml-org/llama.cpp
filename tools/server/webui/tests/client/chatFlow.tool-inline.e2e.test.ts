import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import TestMessagesWrapper from './components/TestMessagesWrapper.svelte';

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

describe('ChatMessages inline tool call rendering (collapse chain)', () => {
	it('shows tool arguments and result inside one reasoning block', async () => {
		const user = msg('u1', 'user', 'Question', null);
		const a1 = msg('a1', 'assistant', '', user.id, {
			thinking: 'think step 1',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '2+2' }) }
				}
			])
		});
		const t1 = msg('t1', 'tool', JSON.stringify({ expression: '2+2', result: '4' }), a1.id, {
			toolCallId: 'call-1'
		});
		const a2 = msg('a2', 'assistant', 'Final answer', t1.id, { thinking: 'think step 2' });

		const messages = [user, a1, t1, a2];

		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages }
		});

		const assistants = container.querySelectorAll('[aria-label="Assistant message with actions"]');
		expect(assistants.length).toBe(1);

		const text = container.textContent || '';
		expect(text).toContain('2+2');
		expect(text).toContain('4');
	});
});
