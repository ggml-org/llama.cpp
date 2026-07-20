import { describe, it, expect } from 'vitest';
import { filterByLeafNodeId } from '$lib/utils/branching';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';

/**
 * Loading a conversation must resolve a path even when currNode is missing
 * (imported records can carry an empty currNode).
 */

const store: DatabaseMessage[] = [];

function add(
	msg: Partial<DatabaseMessage> & { role: MessageRole; type: MessageType },
	parentId: string | null,
	timestamp: number
): DatabaseMessage {
	const node: DatabaseMessage = {
		id: msg.content ? String(msg.content) : `m${store.length + 1}`,
		convId: 'c1',
		content: '',
		toolCalls: '',
		children: [],
		parent: parentId,
		timestamp,
		...msg
	};
	store.push(node);
	if (parentId) {
		const parent = store.find((m) => m.id === parentId);
		if (parent) parent.children = [...parent.children, node.id];
	}
	return node;
}

/** root -> u1 -> a1 -> {u2a -> a2a | u2b -> a2b}: two branches from a1. */
function branchedConversation() {
	store.length = 0;
	add({ role: MessageRole.USER, type: MessageType.ROOT, id: 'root' }, null, 10);
	add({ role: MessageRole.USER, type: MessageType.TEXT, content: 'u1' }, 'root', 20);
	add({ role: MessageRole.ASSISTANT, type: MessageType.TEXT, content: 'a1' }, 'u1', 30);
	add({ role: MessageRole.USER, type: MessageType.TEXT, content: 'u2a' }, 'a1', 40);
	add({ role: MessageRole.ASSISTANT, type: MessageType.TEXT, content: 'a2a' }, 'u2a', 50);
	add({ role: MessageRole.USER, type: MessageType.TEXT, content: 'u2b' }, 'a1', 60);
	add({ role: MessageRole.ASSISTANT, type: MessageType.TEXT, content: 'a2b' }, 'u2b', 70);
}

/** What loadConversation must produce for a given (possibly empty) currNode. */
const resolve = (currNode: string) =>
	(filterByLeafNodeId(store, currNode, false) as DatabaseMessage[]).map((m) => m.id);

describe('conversation load path resolution', () => {
	it('resolves a single branch for a valid currNode', () => {
		branchedConversation();
		expect(resolve('a2a')).toEqual(['u1', 'a1', 'u2a', 'a2a']);
	});

	it('empty currNode still resolves one branch, not every branch', () => {
		branchedConversation();
		const ids = resolve('');
		expect(ids).toEqual(['u1', 'a1', 'u2b', 'a2b']);
		expect(ids).not.toContain('u2a');
		expect(ids).not.toContain('root');
	});

	it('unknown currNode falls back to the latest branch', () => {
		branchedConversation();
		expect(resolve('does-not-exist')).toEqual(['u1', 'a1', 'u2b', 'a2b']);
	});
});
