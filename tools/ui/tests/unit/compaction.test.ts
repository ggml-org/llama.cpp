import { COMPACTION } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';
import { canCompactMessages, compactionCut, sliceAtLastCompaction } from '$lib/utils';
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

function makeCompactionNode(id: string, summary: string, kept?: number): DatabaseMessage {
	return { ...makeMessage(id, MessageRole.USER, summary, MessageType.COMPACTION), kept };
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

describe('sliceAtLastCompaction with kept steps', () => {
	const ids = (messages: DatabaseMessage[]) => messages.map((m) => m.id);

	it('keeps the last steps before the node, a tool result with its call', () => {
		const messages = [
			makeMessage('s1', MessageRole.SYSTEM, 'be brief', MessageType.SYSTEM),
			makeMessage('u1', MessageRole.USER, 'first'),
			makeMessage('a1', MessageRole.ASSISTANT, 'calling'),
			makeMessage('t1', MessageRole.TOOL, 'result'),
			makeMessage('u2', MessageRole.USER, 'second'),
			makeMessage('a2', MessageRole.ASSISTANT, 'answer'),
			makeCompactionNode('c1', 'summary', 3),
			makeMessage('u3', MessageRole.USER, 'next')
		];

		expect(ids(sliceAtLastCompaction(messages))).toEqual([
			's1',
			'c1',
			'a1',
			't1',
			'u2',
			'a2',
			'u3'
		]);
	});

	it('counts the kept steps across an older node, which it drops', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'first'),
			makeMessage('a1', MessageRole.ASSISTANT, 'one'),
			makeCompactionNode('c1', 'first summary', 1),
			makeMessage('u2', MessageRole.USER, 'second'),
			makeMessage('a2', MessageRole.ASSISTANT, 'two'),
			makeCompactionNode('c2', 'second summary', 3),
			makeMessage('u3', MessageRole.USER, 'next')
		];

		expect(ids(sliceAtLastCompaction(messages))).toEqual(['c2', 'a1', 'u2', 'a2', 'u3']);
	});

	it('keeps every step when it keeps more than there are, and none without kept', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'first'),
			makeMessage('a1', MessageRole.ASSISTANT, 'one'),
			makeCompactionNode('c1', 'summary', 9)
		];

		expect(ids(sliceAtLastCompaction(messages))).toEqual(['c1', 'u1', 'a1']);
		expect(
			ids(sliceAtLastCompaction([...messages.slice(0, 2), makeCompactionNode('c1', 's')]))
		).toEqual(['c1']);
	});
});

describe('compactionCut', () => {
	const view = [
		makeMessage('s1', MessageRole.SYSTEM, 'be brief', MessageType.SYSTEM),
		makeMessage('u1', MessageRole.USER, 'first'),
		makeMessage('a1', MessageRole.ASSISTANT, 'calling'),
		makeMessage('t1', MessageRole.TOOL, 'result'),
		makeMessage('u2', MessageRole.USER, 'second'),
		makeMessage('a2', MessageRole.ASSISTANT, 'answer')
	];

	it('cuts before the preserved steps, never between a call and its result', () => {
		expect(compactionCut(view, 2)).toBe(4);
		expect(compactionCut(view, 3)).toBe(2);
		expect(compactionCut(view, 0)).toBe(view.length);
	});

	it('cuts nowhere when no step lies before the preserved ones', () => {
		expect(compactionCut(view, 4)).toBeNull();
		expect(compactionCut(view, 5)).toBeNull();
	});

	it('leaves the preserved steps as the model reads them once compacted', () => {
		const cut = compactionCut(view, 2)!;
		const compacted = sliceAtLastCompaction([...view, makeCompactionNode('c1', 'summary', 2)]);

		expect(compacted.slice(2)).toEqual(view.slice(cut));
		expect(compacted.slice(0, 2).map((m) => m.id)).toEqual(['s1', 'c1']);
	});
});

describe('canCompactMessages', () => {
	it('rejects a branch with no step before the preserved ones', () => {
		expect(canCompactMessages([makeMessage('u1', MessageRole.USER, 'hi')], 1)).toBe(false);
	});

	it('accepts a branch with a step before the preserved ones', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, 'hello')
		];

		expect(canCompactMessages(messages, 1)).toBe(true);
		expect(canCompactMessages(messages, 2)).toBe(false);
	});

	it('rejects a branch with nothing new since the last compaction', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, 'hello'),
			makeCompactionNode('c1', 'summary', 1)
		];

		expect(canCompactMessages(messages, 1)).toBe(false);
	});

	it('accepts a compacted branch once a new step exists past the preserved ones', () => {
		const messages = [
			makeCompactionNode('c1', 'summary'),
			makeMessage('u1', MessageRole.USER, 'next'),
			makeMessage('a1', MessageRole.ASSISTANT, 'reply')
		];

		expect(canCompactMessages(messages, 1)).toBe(true);
	});

	it('accepts a branch whose trailing compaction node is empty', () => {
		const messages = [
			makeMessage('u1', MessageRole.USER, 'hi'),
			makeMessage('a1', MessageRole.ASSISTANT, 'hello'),
			makeCompactionNode('c1', '')
		];

		expect(canCompactMessages(messages, 1)).toBe(true);
	});
});
