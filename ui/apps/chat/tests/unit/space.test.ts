import { parseSpaceArgs, spaceCard, spaceName } from '../../scripts/space.ts';
import { describe, expect, it } from 'vitest';

describe('spaceName', () => {
	it('names the Space after the branch', () => {
		expect(spaceName('feat/voice-mode', 'alice')).toBe('llama-ui-feat-voice-mode');
		expect(spaceName('Fix_the.Thing', 'alice')).toBe('llama-ui-fix-the-thing');
	});

	it('drops the owner the branch starts with', () => {
		expect(spaceName('alice/feat/voice-mode', 'alice')).toBe('llama-ui-feat-voice-mode');
		expect(spaceName('Alice/feat/voice-mode', 'alice')).toBe('llama-ui-feat-voice-mode');
		expect(spaceName('bob/feat/voice-mode', 'alice')).toBe('llama-ui-bob-feat-voice-mode');
	});

	it('keeps a long name within the limit and unique', () => {
		const first = spaceName(`feat/${'a'.repeat(120)}-one`, 'alice');
		const second = spaceName(`feat/${'a'.repeat(120)}-two`, 'alice');

		expect(first).toHaveLength(96);
		expect(first).not.toBe(second);
		expect(first).toMatch(/^[a-z0-9-]+$/);
	});

	it('refuses a branch that leaves no name', () => {
		expect(() => spaceName('///', 'alice')).toThrow();
	});
});

describe('spaceCard', () => {
	const card = spaceCard({ branch: 'feat/voice-mode', repo: 'alice/llama.cpp', sha: '0123456789' });

	it('is a static Space with the headers llama-server serves the UI with', () => {
		expect(card).toContain('sdk: static');
		expect(card).toContain('cross-origin-embedder-policy: require-corp');
		expect(card).toContain('cross-origin-opener-policy: same-origin');
	});

	it('carries the tag the script knows its Spaces by', () => {
		expect(card).toContain('tags:\n  - llama-ui-preview');
	});

	it('says what it was built from', () => {
		expect(card).toContain('`feat/voice-mode`');
		expect(card).toContain('https://github.com/alice/llama.cpp');
		expect(card).toContain('0123456789');
	});
});

describe('parseSpaceArgs', () => {
	it('deploys a public Space by default', () => {
		expect(parseSpaceArgs([])).toMatchObject({
			command: 'deploy',
			ifExists: false,
			visibility: 'public',
			wip: false
		});
	});

	it('reads the command and its options', () => {
		expect(parseSpaceArgs(['rm', '--branch', 'feat/x', '--owner', 'alice'])).toMatchObject({
			branch: 'feat/x',
			command: 'rm',
			owner: 'alice'
		});
		expect(parseSpaceArgs(['--if-exists', '--visibility', 'protected'])).toMatchObject({
			command: 'deploy',
			ifExists: true,
			visibility: 'protected'
		});
	});

	it('refuses what it does not know', () => {
		expect(() => parseSpaceArgs(['launch'])).toThrow();
		expect(() => parseSpaceArgs(['--visibility', 'secret'])).toThrow();
		expect(() => parseSpaceArgs(['--nope'])).toThrow();
	});
});
