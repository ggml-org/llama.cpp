import { beforeAll, describe, expect, it } from 'vitest';
import { zipSync, strToU8 } from 'fflate';
import { MessageRole, MessageType } from '$lib/enums';
import { NEWLINE } from '$lib/constants';
import type { ExportedConversation } from '$lib/types/database';

let conversationsStore: typeof import('$lib/stores/conversations.svelte').conversationsStore;

// node env unit project has no DOM, install a minimal localStorage backed by a
// Map before the store module reads it. Transforming the store takes seconds,
// so import it once for the whole file.
beforeAll(async () => {
	const store = new Map<string, string>();
	const polyfill: Storage = {
		get length() {
			return store.size;
		},
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};
	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;

	({ conversationsStore } = await import('$lib/stores/conversations.svelte'));
}, 30000);

function makeSession(id: string): ExportedConversation {
	return {
		conv: { id, currNode: `${id}-msg`, lastModified: 0, name: `Chat ${id}` },
		messages: [
			{
				id: `${id}-msg`,
				convId: id,
				type: MessageType.TEXT,
				timestamp: 0,
				role: MessageRole.USER,
				content: `hello from ${id}`,
				parent: null,
				children: []
			}
		]
	} as unknown as ExportedConversation;
}

/**
 * `parseImportFile` detects the format from the file contents. iOS has no UTI
 * for `.jsonl`, so the picker cannot filter on it and the filename carries no
 * guarantee: a JSONL export must import under any name.
 */
describe('conversationsStore.parseImportFile', () => {
	it('imports a JSONL export whose name has no meaningful extension', async () => {
		const jsonl = conversationsStore.serializeSessionToJsonl(makeSession('a'));

		const sessions = await conversationsStore.parseImportFile(new File([jsonl], 'export'));

		expect(sessions).toHaveLength(1);
		expect(sessions[0].conv.id).toBe('a');
		expect(sessions[0].messages[0].content).toBe('hello from a');
	});

	it('imports several sessions from one JSONL file', async () => {
		const jsonl = [makeSession('a'), makeSession('b')]
			.map((session) => conversationsStore.serializeSessionToJsonl(session))
			.join(NEWLINE);

		const sessions = await conversationsStore.parseImportFile(new File([jsonl], 'export.txt'));

		expect(sessions.map((session) => session.conv.id)).toEqual(['a', 'b']);
	});

	it('imports a ZIP archive whose name has no meaningful extension', async () => {
		const zipped = zipSync({
			'a.jsonl': strToU8(conversationsStore.serializeSessionToJsonl(makeSession('a'))),
			'b.jsonl': strToU8(conversationsStore.serializeSessionToJsonl(makeSession('b'))),
			'notes.txt': strToU8('ignored')
		});

		const sessions = await conversationsStore.parseImportFile(new File([zipped], 'archive'));

		expect(sessions.map((session) => session.conv.id).sort()).toEqual(['a', 'b']);
	});

	it('imports the legacy JSON array format', async () => {
		const json = JSON.stringify([makeSession('a')], null, 2);

		const sessions = await conversationsStore.parseImportFile(new File([json], 'export.jsonl'));

		expect(sessions).toHaveLength(1);
		expect(sessions[0].conv.id).toBe('a');
	});

	it('imports the legacy JSON single object format', async () => {
		const json = JSON.stringify(makeSession('a'));

		const sessions = await conversationsStore.parseImportFile(new File([json], 'export'));

		expect(sessions).toHaveLength(1);
		expect(sessions[0].conv.id).toBe('a');
	});

	it('rejects a file that holds neither format', async () => {
		await expect(
			conversationsStore.parseImportFile(new File(['not an export'], 'export.jsonl'))
		).rejects.toThrow();
	});
});
