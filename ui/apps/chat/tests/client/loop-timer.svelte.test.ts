// Guards the loop timer: a tool row times its own loop of calls, not the whole
// task. While a reply runs, the last tool row ticks from its first call's start
// ( a single call reads it from context by its id, a group takes it as a prop );
// once done, a group shows the time from its first call's start to its last
// call's end, as the run recorded them.

import ChatMessageAgentTrace from '#lib/components/app/chat/ChatMessages/ChatMessageAgentTrace.svelte';
import { MessageRole, MessageType } from '#lib/enums/index.ts';
import { agenticStore } from '#lib/stores/index.ts';
import type { ChatMessageTimings, DatabaseMessage } from '#lib/types/index.ts';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const CONV_ID = 'conv-timer';
const LIVE_TIMER = '[title="Time since the loop started"]';
const DONE_TIMER = '[title="Time the loop took"]';

type SessionPatch = { toolCallStartedAt: Record<string, number> };
type WithSession = { updateSession(conversationId: string, patch: SessionPatch): void };

function reply(
	calls: number,
	timings?: ChatMessageTimings,
	reasoningContent?: string
): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: CONV_ID,
		id: 'assistant-1',
		parent: null,
		reasoningContent,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		timings,
		toolCalls: JSON.stringify(
			Array.from({ length: calls }, (_, i) => ({
				function: { arguments: `{"path":"file-${i}.ts"}`, name: 'read_file' },
				id: `call-${i}`
			}))
		),
		type: MessageType.TEXT
	};
}

/** Run timings where call i ran from `spans[i][0]` to `spans[i][1]` (ms since epoch). */
function timingsOf(spans: [number, number][]): ChatMessageTimings {
	const llm = { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 };

	return {
		agentic: {
			llm,
			toolCalls: spans.map(([startedAt, endedAt], i) => ({
				duration_ms: endedAt - startedAt,
				endedAt,
				id: `call-${i}`,
				name: 'read_file',
				startedAt,
				success: true
			})),
			toolCallsCount: spans.length,
			toolsMs: 0,
			turns: 1
		}
	};
}

function setCallStarts(toolCallStartedAt: Record<string, number>) {
	// the loop sets these as calls start; there is no public setter
	(agenticStore as unknown as WithSession).updateSession(CONV_ID, { toolCallStartedAt });
}

describe('loop timer', () => {
	beforeEach(() => setCallStarts({ 'call-0': Date.now() - 5000 }));
	afterEach(() => setCallStarts({}));

	it('ticks on a single running call', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: reply(1)
		});

		await tick();

		expect(container.querySelectorAll(LIVE_TIMER)).toHaveLength(1);
		expect(container.querySelector(LIVE_TIMER)?.textContent?.trim()).toMatch(/^\d+s$/);
	});

	it('ticks once on a running group, not on the calls inside it', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: reply(2)
		});

		await tick();

		expect(container.querySelectorAll(LIVE_TIMER)).toHaveLength(1);
		expect(container.querySelector('.tool-call-group')?.querySelector(LIVE_TIMER)).not.toBeNull();
	});

	it('shows the time a finished group took, from its first call to its last', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: false,
			message: reply(
				2,
				timingsOf([
					[1000, 2000],
					[2500, 4500]
				])
			)
		});

		await tick();

		expect(container.querySelectorAll(LIVE_TIMER)).toHaveLength(0);
		expect(container.querySelector(DONE_TIMER)?.textContent?.trim()).toBe('3.5s');
	});

	it('times a group that opens with the reasoning before its first call', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: false,
			message: reply(
				2,
				timingsOf([
					[1000, 2000],
					[2500, 4500]
				]),
				'I should read both files.'
			)
		});

		await tick();

		expect(container.querySelector('.tool-call-group')?.querySelector(DONE_TIMER)).not.toBeNull();
		expect(container.querySelector(DONE_TIMER)?.textContent?.trim()).toBe('3.5s');
	});

	it('shows nothing on a finished reply without call times', async () => {
		const { container } = render(ChatMessageAgentTrace, {
			isLastAssistantMessage: true,
			isStreaming: false,
			message: reply(2)
		});

		await tick();

		expect(container.querySelectorAll(`${LIVE_TIMER}, ${DONE_TIMER}`)).toHaveLength(0);
	});
});
