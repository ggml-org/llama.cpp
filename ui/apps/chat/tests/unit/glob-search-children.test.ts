import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('#lib/services/tools.service.ts', () => ({
	ToolsService: { executeToolRaw: vi.fn() }
}));

import { GlobSearchType } from '#lib/enums/index.ts';
import { ToolsService } from '#lib/services/tools.service.ts';
import { runGlobSearchWithChildren } from '#lib/utils/index.ts';

const mockExecute = vi.mocked(ToolsService.executeToolRaw);

// Distinct roots per test so the module-level search cache never serves a
// prior test's result under the same (type, path, glob, depth) key.
beforeEach(() => {
	mockExecute.mockReset();
});

describe('runGlobSearchWithChildren', () => {
	it('returns ranked outer entries as absolute paths without descending', async () => {
		mockExecute.mockResolvedValueOnce({
			base: '/Users/rootA',
			entries: [
				{ path: 'note.md', type: 'file' },
				{ path: 'src', type: 'dir' }
			]
		});
		const res = await runGlobSearchWithChildren(
			'note',
			'/Users/rootA',
			3,
			50,
			new AbortController().signal
		);

		expect(res.error).toBeUndefined();
		expect(res.entries.map((e) => e.path)).toEqual(['/Users/rootA/note.md', '/Users/rootA/src']);
		expect(res.exactDir).toBeUndefined();
		expect(mockExecute).toHaveBeenCalledTimes(1);
	});

	it('appends a matched directorys children when the query ends with a separator', async () => {
		mockExecute
			.mockResolvedValueOnce({ base: '/Users/rootB', entries: [{ path: 'src', type: 'dir' }] })
			.mockResolvedValueOnce({
				base: '/Users/rootB/src',
				entries: [
					{ path: 'a.txt', type: 'file' },
					{ path: 'sub', type: 'dir' }
				]
			});
		const res = await runGlobSearchWithChildren(
			'/Users/rootB/src/',
			'/Users/rootB',
			3,
			50,
			new AbortController().signal,
			{ descendOnTrailingSeparator: true, type: GlobSearchType.ALL }
		);

		expect(res.error).toBeUndefined();
		expect(res.exactDir).toBe('/Users/rootB/src');
		expect(res.entries.map((e) => e.path)).toEqual([
			'/Users/rootB/src',
			'/Users/rootB/src/a.txt',
			'/Users/rootB/src/sub'
		]);
		expect(mockExecute).toHaveBeenCalledTimes(2);
	});

	it('does not descend without a trailing separator in mention mode', async () => {
		mockExecute.mockResolvedValueOnce({
			base: '/Users/rootC',
			entries: [{ path: 'src', type: 'dir' }]
		});
		const res = await runGlobSearchWithChildren(
			'/Users/rootC/src',
			'/Users/rootC',
			3,
			50,
			new AbortController().signal,
			{ descendOnTrailingSeparator: true, type: GlobSearchType.ALL }
		);

		expect(res.exactDir).toBeUndefined();
		expect(mockExecute).toHaveBeenCalledTimes(1);
	});

	it('descends on an exact directory match in WD mode', async () => {
		mockExecute
			.mockResolvedValueOnce({ base: '/Users/rootD', entries: [{ path: 'src', type: 'dir' }] })
			.mockResolvedValueOnce({
				base: '/Users/rootD/src',
				entries: [{ path: 'a.txt', type: 'file' }]
			});
		const res = await runGlobSearchWithChildren(
			'/Users/rootD/src',
			'/Users/rootD',
			3,
			50,
			new AbortController().signal,
			{ type: GlobSearchType.DIR }
		);

		expect(res.exactDir).toBe('/Users/rootD/src');
		expect(res.entries.map((e) => e.path)).toEqual(['/Users/rootD/src', '/Users/rootD/src/a.txt']);
		expect(mockExecute).toHaveBeenCalledTimes(2);
	});

	it('surfaces a server error without attempting a child walk', async () => {
		mockExecute.mockResolvedValueOnce({ error: 'boom' });
		const res = await runGlobSearchWithChildren(
			'src',
			'/Users/rootE',
			3,
			50,
			new AbortController().signal
		);

		expect(res.error).toBe('boom');
		expect(res.entries).toEqual([]);
		expect(mockExecute).toHaveBeenCalledTimes(1);
	});
});

describe('runGlobSearchWithChildren with a relative path', () => {
	it('lists the folder at the top of the scope before its deeper namesakes', async () => {
		const scope = '/Users/rootF';

		mockExecute.mockImplementation(async (_tool, params) => {
			const { path } = params as { path: string };

			// the direct lookup navigates to the parent; the glob only sees deeper copies
			return path === `${scope}/ui/lib`
				? { base: path, entries: [{ path: 'voice', type: 'dir' }] }
				: { base: scope, entries: [{ path: '.worktrees/a/ui/lib/voice', type: 'dir' }] };
		});

		const res = await runGlobSearchWithChildren(
			'ui/lib/voi',
			scope,
			3,
			50,
			new AbortController().signal,
			{ descendOnTrailingSeparator: true, type: GlobSearchType.ALL }
		);

		expect(res.entries.map((e) => e.path)).toEqual([
			`${scope}/ui/lib/voice`,
			`${scope}/.worktrees/a/ui/lib/voice`
		]);
	});

	it('keeps the deeper matches when the path does not exist at the top', async () => {
		const scope = '/Users/rootG';

		mockExecute.mockImplementation(async (_tool, params) => {
			const { path } = params as { path: string };

			return path === scope
				? { base: scope, entries: [{ path: 'apps/ui/lib/voice', type: 'dir' }] }
				: { error: 'path does not exist or is not a directory' };
		});

		const res = await runGlobSearchWithChildren(
			'ui/lib/voi',
			scope,
			3,
			50,
			new AbortController().signal
		);

		expect(res.error).toBeUndefined();
		expect(res.entries.map((e) => e.path)).toEqual([`${scope}/apps/ui/lib/voice`]);
	});
});

describe('runGlobSearchWithChildren with a rooted path', () => {
	it('looks further down when the folder itself holds no such name', async () => {
		mockExecute
			.mockResolvedValueOnce({ base: '/Users/rootDeep', entries: [] })
			.mockResolvedValueOnce({
				base: '/Users/rootDeep',
				entries: [
					{ path: '.cache/llama.cpp', type: 'dir' },
					{ path: 'git/llama.cpp', type: 'dir' }
				]
			});

		const res = await runGlobSearchWithChildren(
			'~/llama.cpp',
			'/Users/rootDeep/Desktop',
			3,
			50,
			new AbortController().signal
		);

		expect(mockExecute.mock.calls.map((call) => call[1])).toMatchObject([
			{ max_depth: 1, path: '~' },
			{ max_depth: 3, path: '~' }
		]);
		expect(res.entries.map((e) => e.path)).toEqual([
			'/Users/rootDeep/git/llama.cpp',
			'/Users/rootDeep/.cache/llama.cpp'
		]);
	});

	it('stays in the folder when it holds a match', async () => {
		mockExecute.mockResolvedValueOnce({
			base: '/Users/rootNear',
			entries: [{ path: 'whisper.cpp', type: 'dir' }]
		});

		const res = await runGlobSearchWithChildren(
			'~/whisper.cpp',
			'/Users/rootNear/Desktop',
			3,
			50,
			new AbortController().signal,
			{ descendOnTrailingSeparator: true }
		);

		expect(mockExecute).toHaveBeenCalledTimes(1);
		expect(res.entries.map((e) => e.path)).toEqual(['/Users/rootNear/whisper.cpp']);
	});
});

describe('runGlobSearchWithChildren edge cases', () => {
	it('follows a backslash path from a Windows scope', async () => {
		const scope = 'C:/rootH';

		mockExecute.mockImplementation(async (_tool, params) => {
			const { path } = params as { path: string };

			return path === `${scope}/ui/lib`
				? { base: path, entries: [{ path: 'voice', type: 'dir' }] }
				: { base: scope, entries: [] };
		});

		const res = await runGlobSearchWithChildren(
			'ui\\lib\\voi',
			scope,
			3,
			50,
			new AbortController().signal
		);

		expect(res.entries.map((e) => e.path)).toEqual([`${scope}/ui/lib/voice`]);
	});

	it('surfaces the deep search error when the direct lookup finds nothing', async () => {
		const scope = '/Users/rootI';

		mockExecute.mockImplementation(async (_tool, params) => {
			const { path } = params as { path: string };

			return path === scope ? { error: 'boom' } : { base: path, entries: [] };
		});

		const res = await runGlobSearchWithChildren(
			'ui/lib/zzz',
			scope,
			3,
			50,
			new AbortController().signal
		);

		expect(res.error).toBe('boom');
	});
});
