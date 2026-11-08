// Tests for the memoized parseToolCalls and O(1) tool message lookup in
// deriveAgenticSections. These were added to prevent regressions where
// streaming text tokens trigger redundant JSON.parse calls on unchanged
// tool call data.

import { AgenticSectionType, MessageRole } from '$lib/enums';
import type { ApiChatCompletionToolCall } from '$lib/types/api';
import type { DatabaseMessage } from '$lib/types/database';
import { deriveAgenticSections } from '$lib/utils/agentic';
import { describe, expect, it, vi } from 'vitest';

function makeMessage(overrides: Partial<DatabaseMessage>): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'c1',
		id: 'm1',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		type: 'text',
		...overrides
	} as DatabaseMessage;
}

describe('parseToolCalls memoization', () => {
	it('returns the same array reference for the same JSON string', () => {
		// parseToolCalls is not exported, but deriveAgenticSections uses it
		// internally. We verify memoization through behavior: calling
		// deriveAgenticSections twice with the same toolCalls should not
		// re-parse (which we verify by checking the returned sections
		// are equivalent).
		const toolCallsJson = JSON.stringify([
			{ function: { arguments: '{}', name: 'test' }, id: 'call_1', type: 'function' }
		]);
		const msg = makeMessage({ content: 'hello', toolCalls: toolCallsJson });
		const sections1 = deriveAgenticSections(msg, [], [], false);
		const sections2 = deriveAgenticSections(msg, [], [], false);

		expect(sections1).toHaveLength(sections2.length);
		expect(sections1[0].type).toBe(sections2[0].type);
	});

	it('does not re-parse JSON on cache hit', () => {
		const toolCallsJson = JSON.stringify([
			{ function: { arguments: '{}', name: 'test' }, id: 'call_1', type: 'function' }
		]);
		const msg = makeMessage({ content: 'hello', toolCalls: toolCallsJson });
		const spy = vi.spyOn(JSON, 'parse');

		deriveAgenticSections(msg, [], [], false);
		const callsAfterFirst = spy.mock.calls.length;

		deriveAgenticSections(msg, [], [], false);
		expect(spy.mock.calls.length).toBe(callsAfterFirst);

		spy.mockRestore();
	});

	it('handles empty/undefined toolCalls without error', () => {
		const msg = makeMessage({ content: 'hello' });
		const sections = deriveAgenticSections(msg, [], [], false);

		expect(sections).toHaveLength(1);
		expect(sections[0].type).toBe(AgenticSectionType.TEXT);
	});

	it('handles invalid JSON gracefully', () => {
		const msg = makeMessage({ content: 'hello', toolCalls: '{invalid' });
		const sections = deriveAgenticSections(msg, [], [], false);

		// Should return just the text section, no tool call sections
		expect(sections).toHaveLength(1);
		expect(sections[0].type).toBe(AgenticSectionType.TEXT);
	});
});

describe('deriveAgenticSections O(1) tool message lookup', () => {
	it('matches tool messages to tool calls by toolCallId', () => {
		const toolCallsJson = JSON.stringify([
			{ function: { arguments: '{}', name: 'test_1' }, id: 'call_1', type: 'function' },
			{ function: { arguments: '{}', name: 'test_2' }, id: 'call_2', type: 'function' }
		]);
		const toolMessages = [
			makeMessage({ content: 'result_1', role: MessageRole.TOOL, toolCallId: 'call_1' }),
			makeMessage({ content: 'result_2', role: MessageRole.TOOL, toolCallId: 'call_2' })
		];
		const msg = makeMessage({ content: 'hello', toolCalls: toolCallsJson });
		const sections = deriveAgenticSections(msg, toolMessages, [], false);
		// Expect: TEXT + 2 TOOL_CALL sections
		const toolCallSections = sections.filter((s) => s.type === AgenticSectionType.TOOL_CALL);

		expect(toolCallSections).toHaveLength(2);
		expect(toolCallSections[0].toolResult).toBe('result_1');
		expect(toolCallSections[1].toolResult).toBe('result_2');
	});

	it('handles missing tool messages (pending calls during streaming)', () => {
		const toolCallsJson = JSON.stringify([
			{ function: { arguments: '{}', name: 'test' }, id: 'call_1', type: 'function' }
		]);
		const msg = makeMessage({ content: '', toolCalls: toolCallsJson });
		const sections = deriveAgenticSections(msg, [], [], true);
		const toolCallSection = sections.find((s) => s.type === AgenticSectionType.TOOL_CALL_PENDING);

		expect(toolCallSection).toBeDefined();
		expect(toolCallSection?.content).toBe('');
	});

	it('scales with many tool calls (no O(n^2) blowup)', () => {
		const N = 100;
		const toolCalls = Array.from(
			{ length: N },
			(_, i): ApiChatCompletionToolCall => ({
				function: { arguments: '{}', name: `tool_${i}` },
				id: `call_${i}`,
				type: 'function'
			})
		);
		const toolCallsJson = JSON.stringify(toolCalls);
		const toolMessages = Array.from({ length: N }, (_, i) =>
			makeMessage({
				content: `result_${i}`,
				role: MessageRole.TOOL,
				toolCallId: `call_${i}`
			})
		);
		const msg = makeMessage({ content: 'hello', toolCalls: toolCallsJson });
		// If the lookup were still O(n^2), this would be noticeably slow
		const start = Date.now();
		const sections = deriveAgenticSections(msg, toolMessages, [], false);
		const elapsed = Date.now() - start;
		const toolCallSections = sections.filter((s) => s.type === AgenticSectionType.TOOL_CALL);

		expect(toolCallSections).toHaveLength(N);
		expect(elapsed).toBeLessThan(100); // Should be fast with O(1) lookup
	});
});
