import { findCommandToken, takeCommandDismissSnapshot } from '$lib/utils';
import { describe, expect, it } from 'vitest';

describe('findCommandToken', () => {
	it('returns null when the value does not start with a slash', () => {
		expect(findCommandToken('hello /prompt')).toBeNull();
		expect(findCommandToken('')).toBeNull();
		expect(findCommandToken('prompt')).toBeNull();
	});

	it('parses a bare slash', () => {
		expect(findCommandToken('/')).toEqual({ args: '', end: 1, name: '' });
	});

	it('parses a command name with no args', () => {
		expect(findCommandToken('/prompt')).toEqual({ args: '', end: 7, name: 'prompt' });
	});

	it('parses a command name followed by a space', () => {
		expect(findCommandToken('/prompt ')).toEqual({ args: '', end: 8, name: 'prompt' });
	});

	it('parses args after the command name', () => {
		expect(findCommandToken('/prompt rev')).toEqual({ args: 'rev', end: 11, name: 'prompt' });
	});

	it('parses multi-word args', () => {
		expect(findCommandToken('/prompt  review  code ')).toEqual({
			args: ' review  code ',
			end: 22,
			name: 'prompt'
		});
	});

	it('treats the whole run as the name when there is no space', () => {
		expect(findCommandToken('/promptx')).toEqual({ args: '', end: 8, name: 'promptx' });
	});
});

describe('takeCommandDismissSnapshot', () => {
	it('returns null when there is no command token', () => {
		expect(takeCommandDismissSnapshot('hello')).toBeNull();
	});

	it('captures the name and args', () => {
		expect(takeCommandDismissSnapshot('/prompt rev')).toEqual({
			args: 'rev',
			name: 'prompt'
		});
	});
});
