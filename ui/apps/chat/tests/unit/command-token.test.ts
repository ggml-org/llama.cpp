import {
	findCommandToken,
	insertCwdCommand,
	isCommandDismissed,
	locateCwdCommand,
	removeCommandToken,
	replaceWithCwdCommand,
	takeCommandDismissSnapshot
} from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

// the commands that take an argument
const COMMANDS = ['prompt', 'cwd'];

/** The token with the caret at `|` in `text`. */
function tokenAt(text: string, commands: readonly string[] = COMMANDS) {
	const cursor = text.indexOf('|');

	return findCommandToken(text.replace('|', ''), cursor, commands);
}

describe('findCommandToken', () => {
	it('returns null where no command is typed', () => {
		expect(tokenAt('|')).toBeNull();
		expect(tokenAt('prompt|')).toBeNull();
		expect(tokenAt('and/or|')).toBeNull();
		expect(tokenAt('see ~/git|')).toBeNull();
	});

	it('finds a command at the start of the input', () => {
		expect(tokenAt('/|')).toEqual({ args: '', end: 1, name: '', start: 0 });
		expect(tokenAt('/prompt|')).toEqual({ args: '', end: 7, name: 'prompt', start: 0 });
	});

	it('finds a command anywhere in the message', () => {
		expect(tokenAt('fix the build /cw|')).toEqual({ args: '', end: 17, name: 'cw', start: 14 });
		expect(tokenAt('first line\n/mod|')).toEqual({ args: '', end: 15, name: 'mod', start: 11 });
		expect(tokenAt('fix /cw| the build')).toEqual({ args: '', end: 7, name: 'cw', start: 4 });
	});

	it('runs to the end of the word the caret is in', () => {
		expect(tokenAt('/pro|mpt and more')).toEqual({ args: '', end: 7, name: 'prompt', start: 0 });
	});

	it('takes the argument of a known command, single spaces included', () => {
		expect(tokenAt('/prompt rev|')).toEqual({ args: 'rev', end: 11, name: 'prompt', start: 0 });
		expect(tokenAt('fix /cwd ~/My Docs|')).toEqual({
			args: '~/My Docs',
			end: 18,
			name: 'cwd',
			start: 4
		});
	});

	it('reads a path in the argument as the argument, not a command of its own', () => {
		expect(tokenAt('/cwd /usr/lo|')?.name).toBe('cwd');
		expect(tokenAt('/cwd /usr|')?.name).toBe('cwd');
	});

	it('ends the argument at two spaces or a new line', () => {
		expect(tokenAt('/cwd docs  more|')).toBeNull();
		expect(tokenAt('/cwd docs\nmore|')).toBeNull();
	});

	it('leaves the words after a command without an argument to the message', () => {
		expect(tokenAt('/model is broken|')).toBeNull();
		expect(tokenAt('please use /m the small one|')).toBeNull();
		expect(tokenAt('please use /m|')).toEqual({ args: '', end: 13, name: 'm', start: 11 });
	});

	it('takes no argument for an unknown name', () => {
		expect(tokenAt('/promptx|')).toEqual({ args: '', end: 8, name: 'promptx', start: 0 });
		expect(tokenAt('/foo bar|')).toBeNull();
		expect(tokenAt('/cw docs|')).toEqual({ args: 'docs', end: 8, name: 'cw', start: 0 });
	});
});

describe('command dismissal', () => {
	it('holds for the same token only', () => {
		const snapshot = takeCommandDismissSnapshot('hi /prompt rev', 14, COMMANDS);

		expect(snapshot).toEqual({ args: 'rev', name: 'prompt', start: 3 });

		expect(isCommandDismissed(snapshot, { args: 'rev', end: 14, name: 'prompt', start: 3 })).toBe(
			true
		);
		expect(isCommandDismissed(snapshot, { args: 'revi', end: 15, name: 'prompt', start: 3 })).toBe(
			false
		);
		expect(takeCommandDismissSnapshot('hello', 5, COMMANDS)).toBeNull();
	});
});

describe('removeCommandToken', () => {
	it('leaves one space between the words around it', () => {
		expect(removeCommandToken('fix /model the build', { end: 10, start: 4 })).toEqual({
			caret: 4,
			value: 'fix the build'
		});
		expect(removeCommandToken('fix the build /model', { end: 20, start: 14 })).toEqual({
			caret: 13,
			value: 'fix the build'
		});
		expect(removeCommandToken('/model fix it', { end: 6, start: 0 })).toEqual({
			caret: 0,
			value: 'fix it'
		});
		expect(removeCommandToken('/model', { end: 6, start: 0 })).toEqual({ caret: 0, value: '' });
	});
});

describe('the /cwd command', () => {
	it('is written at the caret, set apart by spaces', () => {
		expect(insertCwdCommand('fix the build', 3, '~/git')).toEqual({
			caret: 14,
			start: 4,
			value: 'fix /cwd ~/git the build'
		});
		expect(insertCwdCommand('', 0, '')).toEqual({ caret: 5, start: 0, value: '/cwd ' });
	});

	it('completes a partial token in place', () => {
		expect(replaceWithCwdCommand('fix /cw docs now', { end: 12, start: 4 }, 'docs')).toEqual({
			caret: 13,
			start: 4,
			value: 'fix /cwd docs now'
		});
	});

	it('is found again after an edit elsewhere', () => {
		expect(locateCwdCommand('so, fix /cwd docs now', 4, 'docs', 4)).toEqual({
			end: 17,
			query: 'docs',
			start: 8
		});
		expect(locateCwdCommand('fix the build', 4, 'docs')).toBeNull();
	});

	it('is looked for only where it was or where the edit moved it', () => {
		expect(locateCwdCommand('why does /cwd break', 20, '~/git', -6)).toBeNull();
	});

	it('reads its path to two spaces or the end of the line when it changed', () => {
		expect(locateCwdCommand('/cwd ~/My Docs  more', 0, 'x')).toEqual({
			end: 14,
			query: '~/My Docs',
			start: 0
		});
		expect(locateCwdCommand('/cwd', 0, '')).toEqual({ end: 4, query: '', start: 0 });
	});
});
