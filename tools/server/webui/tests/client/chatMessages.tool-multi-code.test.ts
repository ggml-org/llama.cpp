import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import type { ChatRole, DatabaseMessage } from '$lib/types';
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

describe('ChatMessages with multiple code_interpreter_javascript calls', () => {
	it('does not reuse earlier tool outputs for later tool blocks', async () => {
		settingsStore.config = {
			...SETTING_CONFIG_DEFAULT,
			enableCalculatorTool: false,
			enableCodeInterpreterTool: true
		};

		conversationsStore.activeConversation = {
			id: 'c1',
			name: 'Test',
			currNode: null,
			lastModified: Date.now()
		};

		// Build a single assistant chain with two code_interpreter_javascript calls.
		const user = msg('u1', 'user', 'calc two things', null);
		const a1 = msg('a1', 'assistant', '', user.id, {
			thinking: 'first call',
			toolCalls: JSON.stringify([
				{
					id: 'code-1',
					type: 'function',
					function: {
						name: 'code_interpreter_javascript',
						arguments: JSON.stringify({ code: '1+1' })
					}
				}
			])
		});
		const t1 = msg('t1', 'tool', JSON.stringify({ expression: '1+1', result: '2' }), a1.id, {
			toolCallId: 'code-1'
		});
		const a2 = msg('a2', 'assistant', '', t1.id, {
			thinking: 'second call',
			toolCalls: JSON.stringify([
				{
					id: 'code-2',
					type: 'function',
					function: {
						name: 'code_interpreter_javascript',
						arguments: JSON.stringify({ code: '5*5' })
					}
				}
			])
		});
		// Second tool message is intentionally empty to simulate "pending"
		const t2 = msg('t2', 'tool', '', a2.id, {
			toolCallId: 'code-2'
		});
		const a3 = msg('a3', 'assistant', 'done', t2.id, {});

		const messages = [user, a1, t1, a2, t2, a3];
		conversationsStore.activeMessages = messages;

		const { container } = render(TestMessagesWrapper, {
			target: document.body,
			props: { messages }
		});

		const toolBlocks = container.querySelectorAll('[data-testid="tool-call-block"]');
		expect(toolBlocks.length).toBe(2);

		// First tool shows its result "2"
		expect(toolBlocks[0].textContent || '').toContain('2');

		// Second tool (pending) should NOT show the first result; it should be empty/pending.
		expect(toolBlocks[1].textContent || '').not.toContain('2');
		expect(toolBlocks[1].textContent || '').not.toMatch(/Result\s*2/);
	});
});
