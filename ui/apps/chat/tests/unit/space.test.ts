import {
	isPrunable,
	isSpaceOf,
	parseSpaceArgs,
	pullRequestsDone,
	readSpaceSource,
	spaceCard,
	spaceName
} from '../../scripts/space.ts';
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
	const sha = '0123456789abcdef0123456789abcdef01234567';
	const card = spaceCard({
		branch: 'feat/voice-mode',
		repo: 'alice/llama.cpp',
		server: 'http://localhost:9931',
		sha
	});

	it('is a static Space with the headers llama-server serves the UI with', () => {
		expect(card).toContain('title: Llama Chat preview');
		expect(card).toContain('sdk: static');
		expect(card).toContain('cross-origin-embedder-policy: require-corp');
		expect(card).toContain('cross-origin-opener-policy: same-origin');
	});

	it('takes another title', () => {
		expect(
			spaceCard({ branch: 'x', repo: '', sha: '0', title: 'Llama Connect preview' })
		).toContain('title: Llama Connect preview');
	});

	it('carries the tag the script knows its Spaces by', () => {
		expect(card).toContain('tags:\n  - llama-ui-preview\n');
	});

	it('names the llama-server the preview talks to', () => {
		expect(card).toContain('llama-server on http://localhost:9931');
	});

	it('can be read back for what it was built from', () => {
		expect(readSpaceSource(card)).toEqual({
			branch: 'feat/voice-mode',
			repo: 'alice/llama.cpp',
			sha
		});
		expect(readSpaceSource('---\ntitle: x\n---\n')).toBeNull();
		expect(readSpaceSource('<!-- llama-ui-preview-source {nope} -->')).toBeNull();
	});

	it('says what it was built from', () => {
		expect(card).toContain('`feat/voice-mode`');
		expect(card).toContain('https://github.com/alice/llama.cpp');
		expect(card).toContain('0123456789');
	});
});

describe('isSpaceOf', () => {
	const built = { branch: 'feat/x', repo: 'alice/llama.cpp' };
	const repos = ['ggml-org/llama.cpp', 'Alice/llama.cpp'];

	it('tells the Space of a branch from one that only shares its name', () => {
		expect(isSpaceOf(built, 'feat/x', repos)).toBe(true);
		expect(isSpaceOf(built, 'feat-x', repos)).toBe(false);
		expect(isSpaceOf(built, 'feat/x', ['bob/llama.cpp'])).toBe(false);
	});

	it('takes a repository it does not know for the same one', () => {
		expect(isSpaceOf(built, 'feat/x', [])).toBe(true);
		expect(isSpaceOf({ branch: 'feat/x', repo: '' }, 'feat/x', repos)).toBe(true);
	});
});

describe('isPrunable', () => {
	const source = { branch: 'feat/x', repo: 'alice/llama.cpp', sha: '0' };

	it('takes a Space named after its branch, built from this repository', () => {
		expect(isPrunable('alice/llama-ui-feat-x', 'alice', source, 'Alice/llama.cpp')).toBe(true);
	});

	it('leaves a Space named by hand, of another repository, or without a source', () => {
		expect(isPrunable('alice/llama-chat', 'alice', source, 'alice/llama.cpp')).toBe(false);
		expect(isPrunable('alice/llama-ui-feat-x', 'alice', source, 'bob/llama.cpp')).toBe(false);
		expect(isPrunable('alice/llama-ui-feat-x', 'alice', source, '')).toBe(false);
		expect(isPrunable('alice/llama-ui-feat-x', 'alice', null, 'alice/llama.cpp')).toBe(false);
	});
});

describe('pullRequestsDone', () => {
	const deployedAt = '2026-01-02T00:00:00Z';
	const closedBefore = { closed_at: '2026-01-01T00:00:00Z', state: 'closed' };
	const closedAfter = { closed_at: '2026-01-03T00:00:00Z', state: 'closed' };

	it('keeps a preview while a pull request is open, or when there is none', () => {
		expect(pullRequestsDone([], deployedAt)).toBe(false);
		expect(pullRequestsDone([closedAfter, { closed_at: null, state: 'open' }], deployedAt)).toBe(
			false
		);
	});

	it('lets it go once a pull request closed after its last deploy', () => {
		expect(pullRequestsDone([closedBefore, closedAfter], deployedAt)).toBe(true);
	});

	it('keeps one deployed again after its pull request closed, or of an unknown age', () => {
		expect(pullRequestsDone([closedBefore], deployedAt)).toBe(false);
		expect(pullRequestsDone([closedAfter], undefined)).toBe(false);
	});
});

describe('parseSpaceArgs', () => {
	it('deploys a public Space by default', () => {
		expect(parseSpaceArgs([])).toMatchObject({
			command: 'deploy',
			ifExists: false,
			server: 'http://localhost:9931',
			upstream: 'ggml-org/llama.cpp',
			visibility: 'public',
			wip: false
		});
	});

	it('takes another server and upstream', () => {
		expect(
			parseSpaceArgs(['--server', 'https://llm.example.com', '--upstream', 'alice/llama.cpp'])
		).toMatchObject({ server: 'https://llm.example.com', upstream: 'alice/llama.cpp' });
		expect(() => parseSpaceArgs(['--server', 'localhost:9931'])).toThrow();
		expect(() => parseSpaceArgs(['--server', ''])).toThrow();
		expect(() => parseSpaceArgs(['--upstream', 'llama.cpp'])).toThrow();
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

	it('takes a Space by name', () => {
		expect(parseSpaceArgs(['--space', 'alice/llama-chat-preview']).space).toBe(
			'alice/llama-chat-preview'
		);
		expect(() => parseSpaceArgs(['--space', 'llama-chat-preview'])).toThrow();
	});

	it('refuses what it does not know', () => {
		expect(() => parseSpaceArgs(['launch'])).toThrow();
		expect(() => parseSpaceArgs(['--visibility', 'secret'])).toThrow();
		expect(() => parseSpaceArgs(['--nope'])).toThrow();
	});
});
