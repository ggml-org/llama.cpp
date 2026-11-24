import {
	parseSpaceArgs,
	pullRequestsState,
	readSpaceSource,
	sourceTag,
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
		expect(card).toContain('tags:\n  - llama-ui-preview\n  - llama-ui-src-');
	});

	it('carries the source tag only when it knows the repository', () => {
		expect(card).toContain(`  - ${sourceTag('alice/llama.cpp', 'feat/voice-mode')}`);
		expect(spaceCard({ branch: 'x', repo: '', sha })).not.toContain('llama-ui-src-');
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

describe('sourceTag', () => {
	it('is what the upstream workflow computes: sha1 of "<repo, lowercase>:<branch>"', () => {
		// printf '%s:%s' alice/llama.cpp feat/voice-mode | sha1sum | cut -c1-12
		expect(sourceTag('Alice/llama.cpp', 'feat/voice-mode')).toBe('llama-ui-src-5e796d415769');
	});

	it('keeps the branch as it is', () => {
		expect(sourceTag('alice/llama.cpp', 'Feat/X')).not.toBe(sourceTag('alice/llama.cpp', 'feat/x'));
	});
});

describe('pullRequestsState', () => {
	it('keeps a preview while a pull request is open, or when there is none', () => {
		expect(pullRequestsState([])).toBe('none');
		expect(pullRequestsState([{ state: 'closed' }, { state: 'open' }])).toBe('open');
	});

	it('lets it go once every pull request is closed or merged', () => {
		expect(pullRequestsState([{ state: 'closed' }, { state: 'closed' }])).toBe('done');
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
