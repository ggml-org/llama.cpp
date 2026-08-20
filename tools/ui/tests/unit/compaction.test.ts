import { COMPACTION } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';
import { canCompactMessages, sliceAtLastCompaction } from '$lib/utils';
import { describe, expect, it } from 'vitest';

const CONV_ID = 'conv';

function makeMessage(
	id: string,
	role: MessageRole,
	content: string,
	type: MessageType = MessageType.TEXT
): DatabaseMessage {
	return {
		children: [],
		content,
		convId: CONV_ID,
		id,
		parent: null,
		role,
		timestamp: 0,
		type
	} as DatabaseMessage;
}

function makeCompactionNode(id: string, summary: string): DatabaseMessage {
	return makeMessage(id, MessageRole.USER, summary, MessageType.COMPACTION);
}

describe('sliceAtLastCompaction', () => {
	it('returns the list unchanged when no compaction node is present', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, 'hello')
		];

		expect(sliceAtLastCompaction(messages)).toBe(messages);
	});

	it('drops messages before the boundary and keeps the system prefix', () => {
		const system = makeMessage('s1', MessageRole.SYSTEM, 'be brief', MessageType.SYSTEM);
		const messages = [
			system,
			makeMessage('u1', MessageRole.USER, 'old question'),
			makeMessage('a1', MessageRole.ASSISTANT, 'old answer'),
			makeCompactionNode('c1', 'summary'),
			makeMessage('u2', MessageRole.USER, 'new question')
		];
		const sliced = sliceAtLastCompaction(messages);

		expect(sliced.map((m) => m.id)).toEqual(['s1', 'c1', 'u2']);
		expect(sliced[0]).toBe(system);
	});

	it('frames the compaction node content without mutating the original', () => {
		const node = makeCompactionNode('c1', 'summary text');
		const messages = [makeMessage('u1', MessageRole.USER, 'question'), node];
		const sliced = sliceAtLastCompaction(messages);
		const framed = COMPACTION.CONTEXT_TEMPLATE.replace(
			COMPACTION.SUMMARY_PLACEHOLDER,
			'summary text'
		);

		expect(sliced[0].content).toBe(framed);
		expect(sliced[0].role).toBe(MessageRole.USER);
		expect(node.content).toBe('summary text');
	});

	it('slices at the last boundary when the branch was compacted twice', () => {
		const messages = [
			makeCompactionNode('c1', 'first summary'),
			makeMessage('u1', MessageRole.USER, 'question'),
			makeMessage('a1', MessageRole.ASSISTANT, 'answer'),
			makeCompactionNode('c2', 'second summary'),
			makeMessage('u2', MessageRole.USER, 'follow-up')
		];
		const sliced = sliceAtLastCompaction(messages);

		expect(sliced.map((m) => m.id)).toEqual(['c2', 'u2']);
	});

	it('ignores an empty node left by an interrupted compaction', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'question'),
			makeMessage('a1', MessageRole.ASSISTANT, 'answer'),
			makeCompactionNode('c1', '')
		];

		expect(sliceAtLastCompaction(messages)).toBe(messages);
	});
});

describe('canCompactMessages', () => {
	it('rejects a branch with no assistant turn', () => {
		expect(canCompactMessages([makeMessage('u1', MessageRole.USER, 'hi')])).toBe(false);
	});

	it('accepts a branch with a non-empty assistant turn', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, 'hello')
		];

		expect(canCompactMessages(messages)).toBe(true);
	});

	it('rejects a branch whose only assistant turn is empty', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, '   ')
		];

		expect(canCompactMessages(messages)).toBe(false);
	});

	it('rejects a branch with nothing after the last compaction node', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, 'hello'),
			makeCompactionNode('c1', 'summary')
		];

		expect(canCompactMessages(messages)).toBe(false);
	});

	it('accepts a compacted branch once a new assistant turn exists', () => {
		const messages = [
			makeCompactionNode('c1', 'summary'),
			makeMessage('u1', MessageRole.USER, 'next'),
			makeMessage('a1', MessageRole.ASSISTANT, 'reply')
		];

		expect(canCompactMessages(messages)).toBe(true);
	});

	it('accepts a branch whose trailing compaction node is empty', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, 'hello'),
			makeCompactionNode('c1', '')
		];

		expect(canCompactMessages(messages)).toBe(true);
	});
});
