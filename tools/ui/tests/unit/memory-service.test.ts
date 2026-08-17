import type { DatabaseMemoryEntry } from '$lib/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const TEST_GROUPS = 'areas, people, topics';
const TEST_LIMIT_BYTES = 49152;
const TEST_ENTRY = 'areas/test-entry';
const OTHER_ENTRY = 'topics/other';
const ABSENT_ENTRY = 'areas/absent';
const BAD_GROUP_ENTRY = 'nope/x';
const BAD_SLUG_ENTRY = 'areas/UPPER';
const TRAVERSAL_ENTRY = 'areas/../evil';
const UNCONFIGURED_ENTRY = 'extra/side-notes';
const TEST_DESCRIPTION = 'A test entry';
const OTHER_DESCRIPTION = 'Another description';
const RAW_DESCRIPTION = 'line one\n\tline   two';
const FOLDED_DESCRIPTION = 'line one line two';
const FIRST_LINE = 'first line';
const SECOND_LINE = 'second line';
const DOLLAR_PATTERNS = 'sub(/a/, "$&") and $` and $\' and $$';
const entries = vi.hoisted(() => new Map<string, object>());

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		clearMemoryEntries: async () => {
			entries.clear();
		},
		deleteMemoryEntry: async (name: string) => entries.delete(name),
		getMemoryEntries: async () =>
			[...entries.values()].sort((a, b) =>
				(a as { name: string }).name < (b as { name: string }).name ? -1 : 1
			),
		getMemoryEntry: async (name: string) => entries.get(name),
		memoryTransaction: async <T>(scope: () => Promise<T>) => scope(),
		putMemoryEntry: async (entry: { name: string }) => {
			entries.set(entry.name, entry);
		}
	}
}));

vi.mock('$lib/stores/settings.svelte', () => ({
	settingsStore: {
		get config() {
			return { memoryEntryLimitBytes: TEST_LIMIT_BYTES, memoryGroups: TEST_GROUPS };
		}
	}
}));

import { MEMORY_DONE, MEMORY_EMPTY_INDEX, NEWLINE } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import {
	memoryByteLength,
	MemoryService,
	serializeMemoryEntry
} from '$lib/services/memory.service';

function storedEntry(name: string): DatabaseMemoryEntry {
	return entries.get(name) as DatabaseMemoryEntry;
}

async function open(names?: string[]) {
	return MemoryService.executeTool(BuiltInTool.BROWSER_MEMORY_OPEN, names ? { names } : {});
}

async function write(args: Record<string, unknown>) {
	return MemoryService.executeTool(BuiltInTool.BROWSER_MEMORY_WRITE, args);
}

async function drop(name: string) {
	return MemoryService.executeTool(BuiltInTool.BROWSER_MEMORY_DROP, { name });
}

describe('MemoryService', () => {
	beforeEach(() => {
		entries.clear();
	});

	it('creates an entry and reports its index line', async () => {
		const result = await write({
			description: TEST_DESCRIPTION,
			name: TEST_ENTRY,
			new_str: FIRST_LINE
		});

		expect(result.isError).toBe(false);
		expect(result.content.startsWith(`Created ${TEST_ENTRY}`)).toBe(true);
		expect(result.content).toContain(TEST_DESCRIPTION);
		expect(storedEntry(TEST_ENTRY).body).toBe(`${FIRST_LINE}${NEWLINE}`);
	});

	it('refuses creating without a description', async () => {
		const result = await write({ name: TEST_ENTRY, new_str: FIRST_LINE });

		expect(result.isError).toBe(true);
		expect(result.content).toContain('needs a description');
	});

	it('appends on its own line', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });
		await write({ name: TEST_ENTRY, new_str: SECOND_LINE });

		expect(storedEntry(TEST_ENTRY).body).toBe(`${FIRST_LINE}${NEWLINE}${SECOND_LINE}${NEWLINE}`);
	});

	it('removes a string along with the line break that follows it', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });
		await write({ name: TEST_ENTRY, new_str: SECOND_LINE });

		const result = await write({ name: TEST_ENTRY, old_str: FIRST_LINE });

		expect(result.isError).toBe(false);
		expect(storedEntry(TEST_ENTRY).body).toBe(`${SECOND_LINE}${NEWLINE}`);
	});

	it('splices dollar patterns verbatim', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		const result = await write({ name: TEST_ENTRY, new_str: DOLLAR_PATTERNS, old_str: FIRST_LINE });

		expect(result.isError).toBe(false);
		expect(storedEntry(TEST_ENTRY).body).toBe(`${DOLLAR_PATTERNS}${NEWLINE}`);
	});

	it('refuses a duplicate string with its count', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });
		await write({ name: TEST_ENTRY, new_str: FIRST_LINE });

		const result = await write({ name: TEST_ENTRY, new_str: SECOND_LINE, old_str: FIRST_LINE });

		expect(result.isError).toBe(true);
		expect(result.content).toContain('2 times');
		expect(storedEntry(TEST_ENTRY).body).toBe(`${FIRST_LINE}${NEWLINE}${FIRST_LINE}${NEWLINE}`);
	});

	it('refuses replacing in a missing entry', async () => {
		const result = await write({ name: TEST_ENTRY, new_str: SECOND_LINE, old_str: FIRST_LINE });

		expect(result.isError).toBe(true);
		expect(result.content).toContain('nothing to replace');
	});

	it('reports nothing to do on an empty edit', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		const result = await write({ name: TEST_ENTRY });

		expect(result.isError).toBe(true);
		expect(result.content).toContain('Nothing to do');
	});

	it('reports a description refresh, and only a real one', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		const refreshed = await write({ description: OTHER_DESCRIPTION, name: TEST_ENTRY });

		expect(refreshed.isError).toBe(false);
		expect(refreshed.content).toContain(MEMORY_DONE.DESCRIPTION_REFRESHED);

		const unchanged = await write({ description: OTHER_DESCRIPTION, name: TEST_ENTRY });

		expect(unchanged.isError).toBe(true);
		expect(unchanged.content).toContain('Nothing to do');
	});

	it('folds a description onto one line', async () => {
		await write({ description: RAW_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		expect(storedEntry(TEST_ENTRY).description).toBe(FOLDED_DESCRIPTION);
	});

	it('rejects an unknown group, a bad slug and a traversal', async () => {
		const badGroup = await write({
			description: TEST_DESCRIPTION,
			name: BAD_GROUP_ENTRY,
			new_str: FIRST_LINE
		});
		const badSlug = await write({
			description: TEST_DESCRIPTION,
			name: BAD_SLUG_ENTRY,
			new_str: FIRST_LINE
		});
		const traversal = await write({
			description: TEST_DESCRIPTION,
			name: TRAVERSAL_ENTRY,
			new_str: FIRST_LINE
		});

		expect(badGroup.isError).toBe(true);
		expect(badGroup.content).toContain('Unknown group');
		expect(badSlug.isError).toBe(true);
		expect(traversal.isError).toBe(true);
	});

	it('refuses a write past the size cap and keeps the previous content', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		const oversized = 'x'.repeat(TEST_LIMIT_BYTES);
		const result = await write({ name: TEST_ENTRY, new_str: oversized });

		expect(result.isError).toBe(true);
		expect(result.content).toContain(`over the ${TEST_LIMIT_BYTES} limit`);
		expect(storedEntry(TEST_ENTRY).body).toBe(`${FIRST_LINE}${NEWLINE}`);
	});

	it('renders the index and the empty message', async () => {
		expect((await open()).content).toBe(MEMORY_EMPTY_INDEX);

		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });
		await write({ description: OTHER_DESCRIPTION, name: OTHER_ENTRY, new_str: SECOND_LINE });

		const index = (await open()).content;

		expect(index.split(NEWLINE)).toHaveLength(2);
		expect(index).toContain(TEST_ENTRY);
		expect(index).toContain(OTHER_ENTRY);
	});

	it('opens named entries and flags a fully failed open', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		const mixed = await open([TEST_ENTRY, ABSENT_ENTRY]);

		expect(mixed.isError).toBe(false);
		expect(mixed.content).toContain(`[${TEST_ENTRY}]`);
		expect(mixed.content).toContain(`${ABSENT_ENTRY}: no such entry`);

		const failed = await open([ABSENT_ENTRY]);

		expect(failed.isError).toBe(true);
	});

	it('drops an entry once', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		const first = await drop(TEST_ENTRY);
		const second = await drop(TEST_ENTRY);

		expect(first.isError).toBe(false);
		expect(first.content).toBe(`Dropped ${TEST_ENTRY}`);
		expect(second.isError).toBe(true);
	});

	it('counts bytes on the serialized form in UTF-8', async () => {
		const utf8Body = 'cafe: café → 你好';
		const serialized = serializeMemoryEntry(TEST_DESCRIPTION, utf8Body);

		expect(memoryByteLength(serialized)).toBeGreaterThan(serialized.length);

		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: utf8Body });

		const result = await open([TEST_ENTRY]);

		expect(result.content).toContain(
			`size: ${memoryByteLength(serialized)} of ${TEST_LIMIT_BYTES} bytes`
		);
	});

	it('exports every entry and reimports them untouched', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });
		await write({ description: OTHER_DESCRIPTION, name: OTHER_ENTRY, new_str: SECOND_LINE });

		const exported = await MemoryService.exportEntries();

		expect(exported.entries).toHaveLength(2);

		entries.clear();

		const result = await MemoryService.importEntries(exported);

		expect(result).toEqual({ imported: 2, skipped: 0 });
		expect(storedEntry(TEST_ENTRY)).toEqual(exported.entries.find((e) => e.name === TEST_ENTRY));
	});

	it('leaves an existing entry untouched on import', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });

		const exported = await MemoryService.exportEntries();

		await write({ name: TEST_ENTRY, new_str: SECOND_LINE });

		const result = await MemoryService.importEntries(exported);

		expect(result).toEqual({ imported: 0, skipped: 1 });
		expect(storedEntry(TEST_ENTRY).body).toBe(`${FIRST_LINE}${NEWLINE}${SECOND_LINE}${NEWLINE}`);
	});

	it('refuses a malformed memory file whole', async () => {
		await expect(MemoryService.importEntries(null)).rejects.toThrow('missing entries');
		await expect(MemoryService.importEntries({})).rejects.toThrow('missing entries');

		const malformed = {
			entries: [
				{ body: FIRST_LINE, description: TEST_DESCRIPTION, name: TEST_ENTRY, updated: Date.now() },
				{ body: FIRST_LINE, description: '', name: OTHER_ENTRY, updated: Date.now() }
			]
		};

		await expect(MemoryService.importEntries(malformed)).rejects.toThrow('malformed entry');
		expect(entries.size).toBe(0);
	});

	it('keeps an entry of an unconfigured group listed, readable and droppable, but not writable', async () => {
		const frozen = {
			body: `${FIRST_LINE}${NEWLINE}`,
			description: TEST_DESCRIPTION,
			name: UNCONFIGURED_ENTRY,
			updated: Date.now()
		};

		entries.set(UNCONFIGURED_ENTRY, frozen);

		expect((await open()).content).toContain(UNCONFIGURED_ENTRY);

		const opened = await open([UNCONFIGURED_ENTRY]);

		expect(opened.isError).toBe(false);
		expect(opened.content).toContain(FIRST_LINE);

		const written = await write({ name: UNCONFIGURED_ENTRY, new_str: SECOND_LINE });

		expect(written.isError).toBe(true);
		expect(written.content).toContain('Unknown group');
		expect(storedEntry(UNCONFIGURED_ENTRY)).toEqual(frozen);

		const dropped = await drop(UNCONFIGURED_ENTRY);

		expect(dropped.isError).toBe(false);
		expect(dropped.content).toBe(`Dropped ${UNCONFIGURED_ENTRY}`);
		expect(entries.has(UNCONFIGURED_ENTRY)).toBe(false);
	});

	it('deletes one entry by name and clears them all', async () => {
		await write({ description: TEST_DESCRIPTION, name: TEST_ENTRY, new_str: FIRST_LINE });
		await write({ description: OTHER_DESCRIPTION, name: OTHER_ENTRY, new_str: SECOND_LINE });

		expect(await MemoryService.deleteEntry(TEST_ENTRY)).toBe(true);
		expect(await MemoryService.deleteEntry(TEST_ENTRY)).toBe(false);
		expect(await MemoryService.listEntries()).toHaveLength(1);

		await MemoryService.clearEntries();

		expect(await MemoryService.listEntries()).toHaveLength(0);
	});
});
