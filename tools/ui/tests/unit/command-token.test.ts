import { describe, expect, it } from 'vitest';
import { findCommandToken, takeCommandDismissSnapshot } from '$lib/utils';

describe('findCommandToken', () => {
	it('returns null when the value does not start with a slash', () => {
		expect(findCommandToken('hello /prompt')).toBeNull();
		expect(findCommandToken('')).toBeNull();
		expect(findCommandToken('prompt')).toBeNull();
	});

	it('parses a bare slash', () => {
		expect(findCommandToken('/')).toEqual({ name: '', args: '', end: 1 });
	});

	it('parses a command name with no args', () => {
		expect(findCommandToken('/prompt')).toEqual({ name: 'prompt', args: '', end: 7 });
	});

	it('parses a command name followed by a space', () => {
		expect(findCommandToken('/prompt ')).toEqual({ name: 'prompt', args: '', end: 8 });
	});

	it('parses args after the command name', () => {
		expect(findCommandToken('/prompt rev')).toEqual({ name: 'prompt', args: 'rev', end: 11 });
	});

	it('parses multi-word args', () => {
		expect(findCommandToken('/prompt  review  code ')).toEqual({
			name: 'prompt',
			args: ' review  code ',
			end: 22
		});
	});

	it('treats the whole run as the name when there is no space', () => {
		expect(findCommandToken('/promptx')).toEqual({ name: 'promptx', args: '', end: 8 });
	});
});

describe('takeCommandDismissSnapshot', () => {
	it('returns null when there is no command token', () => {
		expect(takeCommandDismissSnapshot('hello')).toBeNull();
	});

	it('captures the name and args', () => {
		expect(takeCommandDismissSnapshot('/prompt rev')).toEqual({
			name: 'prompt',
			args: 'rev'
		});
	});
});
