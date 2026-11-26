import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('#lib/services/tools.service.ts', () => ({
	ToolsService: { executeToolRaw: vi.fn() }
}));

import { SEARCH } from '#lib/constants/index.ts';
import { ToolsService } from '#lib/services/tools.service.ts';
import { runFolderSearch } from '#lib/utils/index.ts';

const mockExecute = vi.mocked(ToolsService.executeToolRaw);

interface Folders {
	[depth: number]: string[];
}

/** Serve `folders[max_depth]` for the globs that carry `needle`, and nothing for the rest. */
function serve(home: string, needle: string, folders: Folders) {
	mockExecute.mockImplementation(async (_tool, params) => {
		const { include, max_depth: depth, path } = params as Record<string, string | number>;
		const paths = path === home && String(include).includes(needle) ? (folders[+depth] ?? []) : [];

		return { base: String(path), entries: paths.map((p) => ({ path: p, type: 'dir' })) };
	});
}

const signal = () => new AbortController().signal;

// Distinct homes per test so the module-level search cache never serves a
// prior test's result under the same (type, path, glob, depth) key.
beforeEach(() => {
	mockExecute.mockReset();
});

describe('runFolderSearch', () => {
	it('finds a plain name below home, nearest folders first', async () => {
		const home = '/Users/deep';

		serve(home, '[lL][lL][aA][mM][aA]', {
			[SEARCH.MAX_DEPTH]: ['git/llama.cpp', 'git/work/tools/llama-bench'],
			[SEARCH.SHALLOW_DEPTH]: ['git/llama.cpp']
		});

		const partial = vi.fn();
		const res = await runFolderSearch('llama', home, signal(), partial);

		expect(partial.mock.calls[0][0].entries.map((e: { path: string }) => e.path)).toEqual([
			`${home}/git/llama.cpp`
		]);
		expect(res.entries.map((e) => e.path)).toEqual([
			`${home}/git/llama.cpp`,
			`${home}/git/work/tools/llama-bench`
		]);
	});

	it('navigates a query with a path in it', async () => {
		const home = '/Users/nav';

		mockExecute.mockResolvedValueOnce({
			base: `${home}/git`,
			entries: [{ path: 'llama-ui', type: 'dir' }]
		});

		const res = await runFolderSearch('git/llama', home, signal());

		expect(mockExecute.mock.calls[0][1]).toMatchObject({
			max_depth: SEARCH.PATH_NAV_MAX_DEPTH,
			path: `${home}/git`
		});
		expect(res.entries.map((e) => e.path)).toEqual([`${home}/git/llama-ui`]);
	});

	it('allows for a typo when nothing matches exactly', async () => {
		const home = '/Users/typo';

		// only the looser typo globs carry this fragment
		serve(home, '[lL][lL]*[aA]', {
			[SEARCH.FUZZY_MAX_DEPTH]: ['git/llama.cpp', 'git/all-data']
		});

		const res = await runFolderSearch('llana', home, signal());

		expect(res.entries.map((e) => e.path)).toEqual([`${home}/git/llama.cpp`]);
	});

	it('finds nothing for a name that is more than a typo away', async () => {
		const home = '/Users/none';

		serve(home, 'never', {});

		const res = await runFolderSearch('zzzzz', home, signal());

		expect(res.entries).toEqual([]);
		expect(res.error).toBeUndefined();
	});
});
