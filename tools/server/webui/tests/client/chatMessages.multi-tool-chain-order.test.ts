import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import TestMessagesWrapper from './components/TestMessagesWrapper.svelte';

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
	role: 'user' | 'assistant' | 'tool' | 'system',
	content: string,
	parent: string | null,
	timestamp: number,
	extra: Partial<DatabaseMessage> = {}
): DatabaseMessage => ({
	id,
	convId: 'c1',
	type: role === 'tool' ? 'tool' : 'text',
	role,
	content,
	thinking: '',
	toolCalls: '',
	parent: parent ?? '-1',
	children: [],
	timestamp,
	...extra
});

describe('ChatMessages multi-tool chaining and ordering', () => {
	it('keeps all tool results visible when tool messages are parent-chained', async () => {
		const ts = Date.now();
		const user = msg('u1', 'user', 'Use two tools', null, ts);
		const a1 = msg('a1', 'assistant', '', user.id, ts + 1, {
			thinking: 'planning',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '2+2' }) }
				},
				{
					id: 'call-2',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '10-3' }) }
				}
			])
		});
		// Parent-chained tool messages (the branch that should be persisted and reloaded correctly)
		const t1 = msg('t1', 'tool', JSON.stringify({ result: '4' }), a1.id, ts + 2, {
			toolCallId: 'call-1'
		});
		const t2 = msg('t2', 'tool', JSON.stringify({ result: '7' }), t1.id, ts + 3, {
			toolCallId: 'call-2'
		});
		const a2 = msg('a2', 'assistant', 'done', t2.id, ts + 4);

		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages: [user, a1, t1, t2, a2] }
		});

		await waitForText(container, 'done');

		const text = container.textContent || '';
		expect(text).toContain('2+2');
		expect(text).toContain('10-3');
		expect(text).toContain('4');
		expect(text).toContain('7');
		expect(text).toContain('done');
	});

	it('renders deterministic reasoning/tool order when timestamps tie and input is shuffled', async () => {
		const ts = Date.now();
		const user = msg('u1', 'user', 'Question', null, ts);
		const a1 = msg('a1', 'assistant', '', user.id, ts, {
			thinking: 'reason-step-1',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '1+1' }) }
				}
			])
		});
		const t1 = msg('t1', 'tool', JSON.stringify({ result: '2' }), a1.id, ts, {
			toolCallId: 'call-1'
		});
		const a2 = msg('a2', 'assistant', 'final-answer', t1.id, ts, {
			thinking: 'reason-step-2'
		});

		// Intentionally shuffled; deterministic sort should still render the same chain order.
		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages: [a2, t1, user, a1] }
		});

		await waitForText(container, 'final-answer');

		const text = container.textContent || '';
		const idxReason1 = text.indexOf('reason-step-1');
		const idxTool = text.indexOf('1+1');
		const idxReason2 = text.indexOf('reason-step-2');
		const idxFinal = text.indexOf('final-answer');

		expect(idxReason1).toBeGreaterThanOrEqual(0);
		expect(idxTool).toBeGreaterThan(idxReason1);
		expect(idxReason2).toBeGreaterThan(idxTool);
		expect(idxFinal).toBeGreaterThan(idxReason2);
	});

	it('does not hoist later thinking above earlier visible output when phases interleave', async () => {
		const ts = Date.now();
		const user = msg('u1', 'user', 'Interleave phases', null, ts);
		const a1 = msg('a1', 'assistant', 'outside-1', user.id, ts + 1, {
			thinking: 'reason-1',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '3*3' }) }
				}
			])
		});
		const t1 = msg('t1', 'tool', JSON.stringify({ result: '9' }), a1.id, ts + 2, {
			toolCallId: 'call-1'
		});
		const a2 = msg('a2', 'assistant', '', t1.id, ts + 3, {
			thinking: 'reason-2',
			toolCalls: JSON.stringify([
				{
					id: 'call-2',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '5+2' }) }
				}
			])
		});
		const t2 = msg('t2', 'tool', JSON.stringify({ result: '7' }), a2.id, ts + 4, {
			toolCallId: 'call-2'
		});
		const a3 = msg('a3', 'assistant', 'outside-2', t2.id, ts + 5);

		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages: [user, a1, t1, a2, t2, a3] }
		});

		await waitForText(container, 'outside-2');

		const text = container.textContent || '';
		const idxReason1 = text.indexOf('reason-1');
		const idxOutside1 = text.indexOf('outside-1');
		const idxReason2 = text.indexOf('reason-2');
		const idxOutside2 = text.indexOf('outside-2');

		expect(idxReason1).toBeGreaterThanOrEqual(0);
		expect(idxOutside1).toBeGreaterThan(idxReason1);
		expect(idxReason2).toBeGreaterThan(idxOutside1);
		expect(idxOutside2).toBeGreaterThan(idxReason2);
	});
});
