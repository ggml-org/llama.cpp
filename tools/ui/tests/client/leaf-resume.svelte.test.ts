// Pure routing contract for waking a turn after a command-only activation:
// assistant leaves continue through the existing continuation machinery,
// tool result and user leaves open a fresh turn, and an empty conversation
// is a no-op.

import { MessageRole } from '$lib/enums';
import { classifyLeafResume } from '$lib/utils';
import { describe, expect, it } from 'vitest';

describe('classifyLeafResume', () => {
	it('continues assistant leaves through the continuation machinery', () => {
		expect(classifyLeafResume(MessageRole.ASSISTANT)).toBe('continue-assistant');
	});

	it('opens a fresh turn after a tool result leaf', () => {
		expect(classifyLeafResume(MessageRole.TOOL)).toBe('fresh-turn');
	});

	it('opens a fresh turn after a user leaf', () => {
		expect(classifyLeafResume(MessageRole.USER)).toBe('fresh-turn');
	});

	it('is a no-op on an empty conversation', () => {
		expect(classifyLeafResume(undefined)).toBe('no-op');
	});
});
