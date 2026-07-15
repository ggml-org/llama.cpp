import { describe, expect, it } from 'vitest';
import { isExitCodeSummaryLine, parseExecShellCommandExitStatus } from '$lib/utils';

describe('parseExecShellCommandExitStatus', () => {
	it('returns undefined when result is empty', () => {
		expect(parseExecShellCommandExitStatus(undefined)).toBeUndefined();
		expect(parseExecShellCommandExitStatus('')).toBeUndefined();
	});

	it('parses a zero-exit summary at end of clean stdout', () => {
		const status = parseExecShellCommandExitStatus('hello world\n[exit code: 0]');
		expect(status).toEqual({
			code: 0,
			timedOut: false,
			rawText: '[exit code: 0]'
		});
	});

	it('parses a non-zero exit summary', () => {
		const status = parseExecShellCommandExitStatus('cargo: error[E0425]\n[exit code: 101]');
		expect(status?.code).toBe(101);
		expect(status?.timedOut).toBe(false);
	});

	it('detects timed-out suffix', () => {
		const status = parseExecShellCommandExitStatus(
			'still building...\n[exit code: -1] [exit due to timed out]'
		);
		expect(status?.code).toBe(-1);
		expect(status?.timedOut).toBe(true);
	});

	it('tolerates trailing whitespace after the tail line', () => {
		const status = parseExecShellCommandExitStatus('[exit code: 0]   \n\n');
		expect(status?.code).toBe(0);
	});

	it('does not match an explanatory mention of "[exit code:" not at end', () => {
		// Any non-trailing occurrence should NOT trigger the badge - we
		// anchor to the absolute end of the string.
		const status = parseExecShellCommandExitStatus(
			'the shell prints [exit code: 0]\nwhen done\nreally done\n'
		);
		expect(status).toBeUndefined();
	});

	it('does not match mid-stream exit lines followed by more output', () => {
		const status = parseExecShellCommandExitStatus('[exit code: 0]\nmore output keeps streaming');
		expect(status).toBeUndefined();
	});
});

describe('isExitCodeSummaryLine', () => {
	const status = parseExecShellCommandExitStatus('hello\n[exit code: 7]');

	it('matches when line trims to the tail text', () => {
		expect(isExitCodeSummaryLine('   [exit code: 7]   ', status)).toBe(true);
	});

	it('does not match unrelated lines', () => {
		expect(isExitCodeSummaryLine('plain output line', status)).toBe(false);
	});

	it('returns false for missing status argument', () => {
		expect(isExitCodeSummaryLine('[exit code: 7]', undefined)).toBe(false);
	});
});
