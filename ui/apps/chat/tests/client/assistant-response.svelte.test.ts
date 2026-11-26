// Guards the two ways a reply renders: a basic chat reply is its reasoning and
// its answer (ChatMessageAssistantResponse), with no trace rows; a reply that
// called tools is an Agent Trace (ChatMessageAgentTrace), with its calls
// grouped. hasAgenticContent is what tells them apart.

import ChatMessageAssistantResponse from '#lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageAssistant/ChatMessageAssistantResponse.svelte';
import ChatMessageAgentTrace from '#lib/components/app/chat/ChatMessages/ChatMessageAgentTrace.svelte';
import { MessageRole, MessageType } from '#lib/enums/index.ts';
import type { DatabaseMessage } from '#lib/types/index.ts';
import { hasAgenticContent } from '#lib/utils/index.ts';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

function reply(overrides: Partial<DatabaseMessage> = {}): DatabaseMessage {
	return {
		children: [],
		content: 'The answer is 42.',
		convId: 'conv-1',
		id: 'assistant-1',
		parent: null,
		reasoningContent: 'Thinking it over.',
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: '',
		type: MessageType.TEXT,
		...overrides
	};
}

const withTools = reply({
	toolCalls: JSON.stringify([
		{ function: { arguments: '{"path":"a.ts"}', name: 'read_file' }, id: 'c1' },
		{ function: { arguments: '{"path":"b.ts"}', name: 'read_file' }, id: 'c2' }
	])
});

describe('assistant reply views', () => {
	it('treats only a reply that called tools as an agent trace', () => {
		expect(hasAgenticContent(reply())).toBe(false);
		expect(hasAgenticContent(withTools)).toBe(true);
	});

	it('renders a basic reply as its reasoning and its answer', async () => {
		const { container } = render(ChatMessageAssistantResponse, { message: reply() });

		await tick();

		expect(container.textContent).toContain('Reasoning');
		// the answer renders as markdown, which settles a moment later
		await expect.poll(() => container.textContent).toContain('The answer is 42.');
		expect(container.querySelector('.tool-call-group')).toBeNull();
	});

	it('renders a reply with tool calls as a grouped trace', async () => {
		const { container } = render(ChatMessageAgentTrace, { message: withTools });

		await tick();

		expect(container.querySelector('.tool-call-group')).not.toBeNull();
		expect(container.textContent).toContain('Read 2 files');
	});
});
