// Sibling-info correctness for buildSiblingInfoMap, including the memoized
// leaf resolution. A wrong leaf id here breaks branch navigation, so the
// deep-chain and multi-branch cases below pin the resolution down.

import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types/database';
import { buildSiblingInfoMap, findLeafNode } from '$lib/utils/branching';
import { describe, expect, it } from 'vitest';

function msg(id: string, parent: string | null, children: string[] = []): DatabaseMessage {
	return {
		children,
		content: '',
		convId: 'c1',
		id,
		parent,
		role: MessageRole.USER,
		timestamp: 0,
		type: MessageType.TEXT
	} as DatabaseMessage;
}

/** root -> m1 -> ... -> m depth, each node with a single child. */
function linearChain(depth: number): DatabaseMessage[] {
	const messages = [msg('m0', null, ['m1'])];

	for (let i = 1; i <= depth; i++) {
		messages.push(msg(`m${i}`, `m${i - 1}`, i < depth ? [`m${i + 1}`] : []));
	}

	return messages;
}

describe('buildSiblingInfoMap', () => {
	it('resolves the deepest leaf for every node of a long single chain', () => {
		const messages = linearChain(50);
		const map = buildSiblingInfoMap(messages);
		const leafId = messages[messages.length - 1].id;

		// every non-root message of the chain is an only child, and its
		// navigation target is the chain's deepest leaf
		for (const m of messages.slice(1)) {
			const info = map.get(m.id);

			expect(info?.totalSiblings).toBe(1);
			expect(info?.siblingIds).toEqual([leafId]);
		}
	});

	it('reports sibling position and leaf targets on a branched tree', () => {
		// m0 -> m1, m4 ; m1 -> m2 ; m2 -> m3, m6 ; m4 -> m5
		const root = msg('m0', null, ['m1', 'm4']);
		const m1 = msg('m1', 'm0', ['m2']);
		const m2 = msg('m2', 'm1', ['m3', 'm6']);
		const m3 = msg('m3', 'm2');
		const m4 = msg('m4', 'm0', ['m5']);
		const m5 = msg('m5', 'm4');
		const m6 = msg('m6', 'm2');
		const map = buildSiblingInfoMap([root, m1, m2, m3, m4, m5, m6]);

		// m1 and m4 share the root as parent; their nav targets are the
		// leaves of their subtrees ( m6 for the first branch, m5 for the second )
		expect(map.get(m1.id)).toMatchObject({
			currentIndex: 0,
			siblingIds: [m6.id, m5.id],
			totalSiblings: 2
		});
		expect(map.get(m4.id)).toMatchObject({
			currentIndex: 1,
			siblingIds: [m6.id, m5.id],
			totalSiblings: 2
		});

		// m3 and m6 are siblings under m2; both are leaves
		expect(map.get(m3.id)?.siblingIds).toEqual([m3.id, m6.id]);
		expect(map.get(m6.id)?.currentIndex).toBe(1);

		// the root has no parent and reports itself
		expect(map.get(root.id)).toMatchObject({
			currentIndex: 0,
			siblingIds: [root.id],
			totalSiblings: 1
		});
	});

	it('agrees with findLeafNode for arbitrary nodes', () => {
		const messages = linearChain(20);
		const leafId = messages[messages.length - 1].id;

		// every node of the chain resolves to the deepest leaf
		for (const m of messages) {
			expect(findLeafNode(messages, m.id), `leaf of ${m.id}`).toBe(leafId);
		}
	});
});
