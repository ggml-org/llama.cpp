// Pins the contract that makes streaming cheap: updateMessageAtIndex mutates the
// existing message object instead of replacing it.
//
// Replacing it (`{ ...old, ...updates }`) changes the array slot, which
// invalidates every consumer that merely walks the list - ChatMessages'
// `displayMessages` rebuilds entries for EVERY message in the conversation. That
// made per-token cost scale with conversation length (1.26ms at 1 prior message
// -> 3.07ms at 40). Mutating in place keeps it flat.

import { MessageRole } from '$lib/enums';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import type { DatabaseMessage } from '$lib/types';
import { describe, expect, it } from 'vitest';

function makeMessage(id: string): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'c1',
		id,
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		type: 'text'
	} as DatabaseMessage;
}

describe('conversationsStore.updateMessageAtIndex', () => {
	it('mutates in place, preserving object identity', () => {
		conversationsStore.activeMessages = [makeMessage('a'), makeMessage('b')];
		const before = conversationsStore.activeMessages[1];

		conversationsStore.updateMessageAtIndex(1, { content: 'hello' });

		expect(conversationsStore.activeMessages[1].content).toBe('hello');
		expect(conversationsStore.activeMessages[1]).toBe(before);

		conversationsStore.activeMessages = [];
	});

	it('leaves other messages and unrelated fields untouched', () => {
		conversationsStore.activeMessages = [makeMessage('a'), makeMessage('b')];
		const untouched = conversationsStore.activeMessages[0];

		conversationsStore.updateMessageAtIndex(1, { content: 'x', model: 'm1' });

		expect(conversationsStore.activeMessages[0]).toBe(untouched);
		expect(conversationsStore.activeMessages[0].content).toBe('');
		expect(conversationsStore.activeMessages[1].model).toBe('m1');
		expect(conversationsStore.activeMessages[1].id).toBe('b');

		conversationsStore.activeMessages = [];
	});

	it('is a no-op for an index of -1 or out of range', () => {
		conversationsStore.activeMessages = [makeMessage('a')];

		expect(() => conversationsStore.updateMessageAtIndex(-1, { content: 'x' })).not.toThrow();
		expect(() => conversationsStore.updateMessageAtIndex(9, { content: 'x' })).not.toThrow();
		expect(conversationsStore.activeMessages[0].content).toBe('');

		conversationsStore.activeMessages = [];
	});
});
