import {
	composeCwdCommandLine,
	cwdCommandCaretOffset,
	findCommandToken,
	parseCwdCommandLine,
	takeCommandDismissSnapshot
} from '#lib/utils/index.ts';
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

describe('parseCwdCommandLine', () => {
	it('returns null without a /cwd command', () => {
		expect(parseCwdCommandLine('hello')).toBeNull();
		expect(parseCwdCommandLine('/cwdx docs')).toBeNull();
		expect(parseCwdCommandLine('/prompt docs')).toBeNull();
	});

	it('splits the path on the first line from the message below it', () => {
		expect(parseCwdCommandLine('/cwd ~/docs\nfix the\nbuild')).toEqual({
			message: 'fix the\nbuild',
			query: '~/docs'
		});
	});

	it('reads a bare command as an empty path', () => {
		expect(parseCwdCommandLine('/cwd')).toEqual({ message: '', query: '' });
		expect(parseCwdCommandLine('/cwd\nhi')).toEqual({ message: 'hi', query: '' });
	});
});

describe('composeCwdCommandLine', () => {
	it('puts the /cwd line above the message', () => {
		expect(composeCwdCommandLine('docs', 'hi')).toBe('/cwd docs\nhi');
		expect(composeCwdCommandLine('docs', '')).toBe('/cwd docs');
	});

	it('round-trips through parseCwdCommandLine', () => {
		expect(parseCwdCommandLine(composeCwdCommandLine('a b', 'c\nd'))).toEqual({
			message: 'c\nd',
			query: 'a b'
		});
	});

	it('places the caret at the end of the path', () => {
		expect(cwdCommandCaretOffset('docs')).toBe('/cwd docs'.length);
	});
});
