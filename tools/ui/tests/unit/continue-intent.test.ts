import { describe, it, expect } from 'vitest';
import { classifyContinueIntent } from '$lib/utils/agentic';
import { ContinueIntentKind, MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types/database';

/**
 * Tests for the Continue button intent classifier.
 *
 * The classifier walks the persisted message history to decide which of three
 * resume paths a Continue click should take:
 *
 *   A. append_text  -> plain text assistant turn, resume with
 *                      continue_final_message.
 *   B. rerun_turn   -> assistant turn with tool_calls but no tool results yet,
 *                      the stream was cut mid turn and the tool_calls are
 *                      unrecoverable as a token level continuation. Drop the
 *                      target and rerun from the previous history.
 *   C. next_turn    -> assistant turn with tool_calls that were already
 *                      resolved by trailing tool results. Hand the history
 *                      back to the agentic loop so it starts the next turn.
 */

let nextId = 0;
function makeMsg(role: MessageRole, opts: Partial<DatabaseMessage> = {}): DatabaseMessage {
	nextId++;
	return {
		id: `msg-${nextId}`,
		convId: 'conv-1',
		type: MessageType.TEXT,
		timestamp: nextId,
		role,
		content: '',
		parent: null,
		children: [],
		...opts
	};
}

function toolCall(id: string, name: string, args: string = '{}'): string {
	return JSON.stringify([{ id, type: 'function', function: { name, arguments: args } }]);
}

describe('classifyContinueIntent', () => {
	it('returns append_text for a plain text assistant turn at the tail', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'hello' }),
			makeMsg(MessageRole.ASSISTANT, { content: 'hi there' })
		];

		const intent = classifyContinueIntent(messages, 1);

		expect(intent).toEqual({ kind: ContinueIntentKind.APPEND_TEXT });
	});

	it('returns append_text for a plain text assistant turn in the middle', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'q1' }),
			makeMsg(MessageRole.ASSISTANT, { content: 'a1' }),
			makeMsg(MessageRole.USER, { content: 'q2' }),
			makeMsg(MessageRole.ASSISTANT, { content: 'a2' })
		];

		expect(classifyContinueIntent(messages, 1)).toEqual({ kind: ContinueIntentKind.APPEND_TEXT });
	});

	it('returns rerun_turn when the assistant has tool_calls without results', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'list files' }),
			makeMsg(MessageRole.ASSISTANT, {
				content: '',
				toolCalls: toolCall('call_1', 'bash_tool', '{"command":"ls"}')
			})
		];

		const intent = classifyContinueIntent(messages, 1);

		expect(intent).toEqual({ kind: ContinueIntentKind.RERUN_TURN, truncateAfter: 0 });
	});

	it('returns next_turn when trailing tool results resolve the tool_calls', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'list files' }),
			makeMsg(MessageRole.ASSISTANT, {
				content: '',
				toolCalls: toolCall('call_1', 'bash_tool')
			}),
			makeMsg(MessageRole.TOOL, { content: 'file1\nfile2', toolCallId: 'call_1' })
		];

		const intent = classifyContinueIntent(messages, 1);

		expect(intent).toEqual({ kind: ContinueIntentKind.NEXT_TURN, truncateAfter: 2 });
	});

	it('next_turn keeps all consecutive trailing tool results, not just one', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'do many things' }),
			makeMsg(MessageRole.ASSISTANT, {
				content: '',
				toolCalls: JSON.stringify([
					{ id: 'call_1', type: 'function', function: { name: 'a', arguments: '{}' } },
					{ id: 'call_2', type: 'function', function: { name: 'b', arguments: '{}' } }
				])
			}),
			makeMsg(MessageRole.TOOL, { content: 'r1', toolCallId: 'call_1' }),
			makeMsg(MessageRole.TOOL, { content: 'r2', toolCallId: 'call_2' })
		];

		const intent = classifyContinueIntent(messages, 1);

		expect(intent).toEqual({ kind: ContinueIntentKind.NEXT_TURN, truncateAfter: 3 });
	});

	it('next_turn stops at the first non tool message after the target', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'go' }),
			makeMsg(MessageRole.ASSISTANT, {
				content: '',
				toolCalls: toolCall('call_1', 'a')
			}),
			makeMsg(MessageRole.TOOL, { content: 'r1', toolCallId: 'call_1' }),
			makeMsg(MessageRole.USER, { content: 'wait' }),
			makeMsg(MessageRole.TOOL, { content: 'late', toolCallId: 'call_1' })
		];

		const intent = classifyContinueIntent(messages, 1);

		// truncateAfter must point at the contiguous tool block, not jump over
		// the user message to grab the dangling late tool.
		expect(intent).toEqual({ kind: ContinueIntentKind.NEXT_TURN, truncateAfter: 2 });
	});

	it('returns append_text when toolCalls is set but parses to empty array', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'q' }),
			makeMsg(MessageRole.ASSISTANT, { content: 'a', toolCalls: '[]' })
		];

		expect(classifyContinueIntent(messages, 1)).toEqual({ kind: ContinueIntentKind.APPEND_TEXT });
	});

	it('returns append_text when toolCalls is malformed JSON', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'q' }),
			makeMsg(MessageRole.ASSISTANT, { content: 'a', toolCalls: '{not json' })
		];

		expect(classifyContinueIntent(messages, 1)).toEqual({ kind: ContinueIntentKind.APPEND_TEXT });
	});

	it('returns append_text defensively when idx points at a non assistant message', () => {
		const messages = [
			makeMsg(MessageRole.USER, { content: 'q' }),
			makeMsg(MessageRole.ASSISTANT, { content: 'a' })
		];

		expect(classifyContinueIntent(messages, 0)).toEqual({ kind: ContinueIntentKind.APPEND_TEXT });
	});

	it('returns append_text defensively when idx is out of bounds', () => {
		const messages = [makeMsg(MessageRole.ASSISTANT, { content: 'a' })];

		expect(classifyContinueIntent(messages, 5)).toEqual({ kind: ContinueIntentKind.APPEND_TEXT });
		expect(classifyContinueIntent([], 0)).toEqual({ kind: ContinueIntentKind.APPEND_TEXT });
	});
});
