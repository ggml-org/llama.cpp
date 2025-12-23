import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import type { ChatRole, DatabaseMessage } from '$lib/types';
import TestMessagesWrapper from './components/TestMessagesWrapper.svelte';

const waitForText = async (container: HTMLElement, text: string, timeoutMs = 2000) => {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if ((container.textContent || '').includes(text)) return;
		await new Promise((resolve) => setTimeout(resolve, 16));
	}
	throw new Error(`Timed out waiting for text: ${text}`);
};

// Utility to build a message quickly
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

describe('ChatMessages inline tool rendering', () => {
	it('collapses reasoning+tool chain and shows arguments and result in one block', async () => {
		// Enable calculator tool (client-side tools)
		settingsStore.config = { ...SETTING_CONFIG_DEFAULT, enableCalculatorTool: true };

		// Conversation context
		conversationsStore.activeConversation = {
			id: 'c1',
			name: 'Test',
			currNode: null,
			lastModified: Date.now()
		};

		// Message chain: user -> assistant(thinking+toolcall) -> tool -> assistant(thinking) -> tool -> assistant(final)
		const user = msg('u1', 'user', 'Question', null);
		const a1 = msg('a1', 'assistant', 'Let me calculate that.', user.id, {
			thinking: 'step1',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '20.25/7.84' }) }
				}
			])
		});
		const t1 = msg(
			't1',
			'tool',
			JSON.stringify({ expression: '20.25/7.84', result: '2.5829', duration_ms: 1234 }),
			a1.id,
			{
				toolCallId: 'call-1'
			}
		);
		const a2 = msg('a2', 'assistant', '', t1.id, {
			thinking: 'step2',
			toolCalls: JSON.stringify([
				{
					id: 'call-2',
					type: 'function',
					function: {
						name: 'calculator',
						arguments: JSON.stringify({ expression: 'log2(2.5829)' })
					}
				}
			])
		});
		const t2 = msg(
			't2',
			'tool',
			JSON.stringify({ expression: 'log2(2.5829)', result: '1.3689', duration_ms: 50 }),
			a2.id,
			{
				toolCallId: 'call-2'
			}
		);
		const a3 = msg('a3', 'assistant', 'About 1.37 stops', t2.id, { thinking: 'final step' });

		const messages = [user, a1, t1, a2, t2, a3];
		conversationsStore.activeMessages = messages;

		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages }
		});

		await waitForText(container, 'Arguments');
		await waitForText(container, 'Let me calculate that.');

		// One assistant card after collapsing the chain
		const assistants = container.querySelectorAll('[aria-label="Assistant message with actions"]');
		expect(assistants.length).toBe(1);

		// Arguments and result should both be visible
		expect(container.textContent).toContain('Arguments');
		expect(container.textContent).toContain('20.25/7.84');
		expect(container.textContent).toContain('1.3689');
		expect(container.textContent).toContain('1.23s');

		// Content produced before the first tool call should not be lost when the chain collapses.
		expect(container.textContent).toContain('Let me calculate that.');
	});

	it('does not render post-reasoning tool calls inside the reasoning block', async () => {
		settingsStore.config = {
			...SETTING_CONFIG_DEFAULT,
			enableCalculatorTool: true,
			showThoughtInProgress: true
		};

		conversationsStore.activeConversation = {
			id: 'c1',
			name: 'Test',
			currNode: null,
			lastModified: Date.now()
		};

		const user = msg('u1', 'user', 'Question', null);
		const a1 = msg('a1', 'assistant', 'Here is the answer (before tool).', user.id, {
			thinking: 'done thinking',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '1+1' }) }
				}
			]),
			// Simulate streaming so the reasoning block is expanded and in-DOM.
			timestamp: 0
		});
		const t1 = msg(
			't1',
			'tool',
			JSON.stringify({ expression: '1+1', result: '2', duration_ms: 10 }),
			a1.id,
			{
				toolCallId: 'call-1'
			}
		);
		const a2 = msg('a2', 'assistant', 'And here is the rest (after tool).', t1.id, {
			timestamp: 0
		});

		const messages = [user, a1, t1, a2];
		conversationsStore.activeMessages = messages;

		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages }
		});

		await waitForText(container, 'Arguments');
		await waitForText(container, 'Here is the answer (before tool).');
		await waitForText(container, 'And here is the rest (after tool).');

		const assistant = container.querySelector('[aria-label="Assistant message with actions"]');
		expect(assistant).toBeTruthy();

		// Tool call should exist overall...
		expect(container.querySelectorAll('[data-testid="tool-call-block"]').length).toBe(1);

		// ...but it should not be rendered inside the reasoning collapsible content.
		const reasoningRoot = assistant
			? Array.from(assistant.querySelectorAll('[data-state]')).find((el) =>
					(el.textContent ?? '').includes('Reasoning')
				)
			: null;
		expect(reasoningRoot).toBeTruthy();
		expect(reasoningRoot?.querySelectorAll('[data-testid="tool-call-block"]').length ?? 0).toBe(0);

		// Ordering: pre-tool content -> tool arguments -> post-tool content.
		const fullText = container.textContent ?? '';
		expect(fullText.indexOf('Here is the answer (before tool).')).toBeGreaterThanOrEqual(0);
		expect(fullText.indexOf('Arguments')).toBeGreaterThan(
			fullText.indexOf('Here is the answer (before tool).')
		);
		expect(fullText.indexOf('And here is the rest (after tool).')).toBeGreaterThan(
			fullText.indexOf('Arguments')
		);
	});
});
