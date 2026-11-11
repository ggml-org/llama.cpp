// search_chats starts disabled: tool policy is a deny-list, so the key must be added to the global defaults and to every existing conversation

import { DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY, SEARCH_CHATS_TOOL_KEY } from '$lib/constants';
import { DatabaseService } from '$lib/services/database.service';
import { MigrationService } from '$lib/services/migration.service';
import type { ExportedConversation } from '$lib/types/database';
import { withSearchChatsDisabled } from '$lib/utils/search-chats';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const MIGRATION_ID = 'search-chats-default-disabled-v1';

async function runMigration() {
	await MigrationService.getMigrations()
		.find((m) => m.id === MIGRATION_ID)
		?.run();
}

const conversation = (id: string, disabledTools?: string[]) =>
	({
		conv: { currNode: null, disabledTools, id, lastModified: 0, name: id },
		messages: []
	}) as unknown as ExportedConversation;

function storedDefaults(): unknown {
	return JSON.parse(localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY) ?? 'null');
}

describe('search_chats default-disabled migration', () => {
	beforeEach(() => {
		localStorage.removeItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);
	});

	afterEach(async () => {
		const conversations = await DatabaseService.getAllConversations();

		await DatabaseService.bulkDeleteConversations(conversations.map((c) => c.id));
		localStorage.removeItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);
	});

	it('adds the tool to the global defaults, keeping existing entries', async () => {
		localStorage.setItem(
			DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
			JSON.stringify(['server:exec_shell_command'])
		);
		await runMigration();
		expect(storedDefaults()).toEqual(['server:exec_shell_command', SEARCH_CHATS_TOOL_KEY]);
	});

	it('creates the defaults on a fresh install', async () => {
		await runMigration();
		expect(storedDefaults()).toEqual([SEARCH_CHATS_TOOL_KEY]);
	});

	it('disables the tool in every existing conversation', async () => {
		await DatabaseService.importConversations([
			conversation('no-policy'),
			conversation('with-policy', ['browser:run_javascript']),
			conversation('already', [SEARCH_CHATS_TOOL_KEY])
		]);
		await runMigration();

		expect((await DatabaseService.getConversation('no-policy'))?.disabledTools).toEqual([
			SEARCH_CHATS_TOOL_KEY
		]);
		expect((await DatabaseService.getConversation('with-policy'))?.disabledTools).toEqual([
			'browser:run_javascript',
			SEARCH_CHATS_TOOL_KEY
		]);
		expect((await DatabaseService.getConversation('already'))?.disabledTools).toEqual([
			SEARCH_CHATS_TOOL_KEY
		]);
	});

	it('keeps the tool policy of the source chat on fork', async () => {
		await DatabaseService.importConversations([
			{
				conv: {
					currNode: 'm1',
					disabledTools: [SEARCH_CHATS_TOOL_KEY],
					id: 'source',
					lastModified: 0,
					name: 'source'
				},
				messages: [
					{
						children: [],
						content: 'hi',
						convId: 'source',
						id: 'm1',
						parent: null,
						role: 'user',
						timestamp: 0,
						type: 'text'
					}
				]
			} as unknown as ExportedConversation
		]);

		const fork = await DatabaseService.forkConversation('source', 'm1', {
			includeAttachments: false,
			name: 'fork'
		});

		expect(fork.disabledTools).toEqual([SEARCH_CHATS_TOOL_KEY]);
	});
});

describe('withSearchChatsDisabled', () => {
	it('adds the key once, for imported rows with or without a policy', () => {
		expect(withSearchChatsDisabled(undefined)).toEqual([SEARCH_CHATS_TOOL_KEY]);
		expect(withSearchChatsDisabled(['a'])).toEqual(['a', SEARCH_CHATS_TOOL_KEY]);
		expect(withSearchChatsDisabled([SEARCH_CHATS_TOOL_KEY])).toEqual([SEARCH_CHATS_TOOL_KEY]);
	});
});
