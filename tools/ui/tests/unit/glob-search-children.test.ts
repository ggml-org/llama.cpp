import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/tools.service', () => ({
	ToolsService: { executeToolRaw: vi.fn() }
}));

import { ToolsService } from '$lib/services/tools.service';
import { GlobSearchType } from '$lib/enums';
import { runGlobSearchWithChildren } from '$lib/utils';

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
			{ type: GlobSearchType.ALL, descendOnTrailingSeparator: true }
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
			{ type: GlobSearchType.ALL, descendOnTrailingSeparator: true }
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
