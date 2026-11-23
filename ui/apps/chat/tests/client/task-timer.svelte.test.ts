// Guards the live task timer: while a reply runs, the last tool row shows the
// time since the task started. A single call reads it from context by its id;
// a group takes it as a prop; a finished reply shows none.

import ChatMessageAgentTrace from '#lib/components/app/chat/ChatMessages/ChatMessageAgentTrace.svelte';
import { MessageRole, MessageType } from '#lib/enums/index.ts';
import { agenticStore } from '#lib/stores/index.ts';
import type { DatabaseMessage } from '#lib/types/index.ts';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const CONV_ID = 'conv-timer';
const TIMER = '[title="Time since the task started"]';

type SessionPatch = { startedAt: number | null };
type WithSession = { updateSession(conversationId: string, patch: SessionPatch): void };

function reply(calls: number): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: CONV_ID,
		id: 'assistant-1',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: JSON.stringify(
			Array.from({ length: calls }, (_, i) => ({
				function: { arguments: `{"path":"file-${i}.ts"}`, name: 'read_file' },
				id: `call-${i}`
			}))
		),
		type: MessageType.TEXT
	};
}

function setStartedAt(startedAt: number | null) {
	// the loop sets this when a flow starts; there is no public setter
	(agenticStore as unknown as WithSession).updateSession(CONV_ID, { startedAt });
}

describe('task timer', () => {
	beforeEach(() => setStartedAt(Date.now() - 5000));
	afterEach(() => setStartedAt(null));

	it('shows on a single running call', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: reply(1)
		});

		await tick();

		expect(container.querySelectorAll(TIMER)).toHaveLength(1);
		expect(container.querySelector(TIMER)?.textContent?.trim()).toMatch(/^\d+s$/);
	});

	it('shows once on a group, not on the calls inside it', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: reply(2)
		});

		await tick();

		expect(container.querySelectorAll(TIMER)).toHaveLength(1);
		expect(container.querySelector('.tool-call-group')?.querySelector(TIMER)).not.toBeNull();
	});

	it('shows nothing once the reply is not streaming', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: false,
			message: reply(1)
		});

		await tick();

		expect(container.querySelectorAll(TIMER)).toHaveLength(0);
	});
});
