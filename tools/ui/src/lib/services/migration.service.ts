/**
 * Migration Service - Unified data migration hook
 *
 * Centralizes all data migrations (localStorage, IndexedDB, legacy formats) into a single
 * initialization point. Each migration copies data to new format WITHOUT deleting the old.
 *
 * **Architecture:**
 * - Migrations are defined as objects with `id` and `run()` methods
 * - Migration state is tracked in localStorage to avoid re-running
 * - `runAllMigrations()` should be called once at app startup
 * - All migrations are NON-DESTRUCTIVE - legacy data is preserved for downgrade compatibility
 *
 * **Current Migrations:**
 * 1. localStorage prefix: Copy LlamaCppWebui.* → LlamaUi.* (both preserved)
 * 2. IndexedDB database: Copy LlamacppWebui → LlamaUi (both preserved)
 * 3. Legacy message format: Transform in-place (preserves structure, migrates markers)
 * 4. Theme key: Copy standalone `theme` → config object (both preserved)
 */

import Dexie from 'dexie';
import {
	STORAGE_APP_NAME,
	DB_APP_NAME_DEPRECATED,
	CONFIG_LOCALSTORAGE_KEY,
	IDXDB_TABLES,
	IDXDB_STORES,
	NEW_TO_DEPRECATED_MAP
} from '$lib/constants';
import { LEGACY_AGENTIC_REGEX, LEGACY_REASONING_TAGS } from '$lib/constants/agentic';
import { SETTINGS_KEYS } from '$lib/constants/settings-registry';
import { MessageRole } from '$lib/enums';

// Types

interface Migration {
	/** Unique identifier for this migration */
	id: string;
	/** Human-readable description */
	description: string;
	/** Run the migration forward (non-destructive - copies, doesn't delete) */
	run(): Promise<void>;
}

interface MigrationState {
	completed: string[];
	failed: string[];
	lastRun: string;
}

// Constants

const MIGRATION_STATE_KEY = `${STORAGE_APP_NAME}.migration-state`;
const MIGRATION_STATE_VERSION = 1;

// State Management

function getMigrationState(): MigrationState {
	try {
		const raw = localStorage.getItem(MIGRATION_STATE_KEY);
		if (!raw) return { completed: [], failed: [], lastRun: '' };
		const parsed = JSON.parse(raw);
		if (parsed.version !== MIGRATION_STATE_VERSION) {
			return { completed: [], failed: [], lastRun: '' };
		}
		return {
			completed: parsed.completed ?? [],
			failed: parsed.failed ?? [],
			lastRun: parsed.lastRun ?? ''
		};
	} catch {
		return { completed: [], failed: [], lastRun: '' };
	}
}

function saveMigrationState(state: MigrationState): void {
	localStorage.setItem(
		MIGRATION_STATE_KEY,
		JSON.stringify({
			version: MIGRATION_STATE_VERSION,
			...state,
			lastRun: new Date().toISOString()
		})
	);
}

function isMigrationCompleted(id: string): boolean {
	const state = getMigrationState();
	return state.completed.includes(id);
}

function markMigrationCompleted(id: string): void {
	const state = getMigrationState();
	if (!state.completed.includes(id)) {
		state.completed.push(id);
	}
	state.failed = state.failed.filter((f) => f !== id);
	saveMigrationState(state);
}

function markMigrationFailed(id: string): void {
	const state = getMigrationState();
	if (!state.failed.includes(id)) {
		state.failed.push(id);
	}
	saveMigrationState(state);
}

// Migration 1: LocalStorage Key Prefix (Non-Destructive)

const LOCALSTORAGE_MIGRATION_ID = 'localstorage-prefix-v1';

const localStorageMigration: Migration = {
	id: LOCALSTORAGE_MIGRATION_ID,
	description: 'Copy localStorage keys from LlamaCppWebui to LlamaUi prefix (non-destructive)',

	async run(): Promise<void> {
		// Non-destructive: copy to new key, but KEEP the old key
		for (const [newKey, deprecatedKey] of Object.entries(NEW_TO_DEPRECATED_MAP)) {
			// Only migrate if new key doesn't already exist
			const newValue = localStorage.getItem(newKey);
			if (newValue !== null) {
				console.log(`[Migration] localStorage: ${newKey} already exists, skipping`);
				continue;
			}

			const oldValue = localStorage.getItem(deprecatedKey);
			if (oldValue !== null) {
				localStorage.setItem(newKey, oldValue);
				// Keep old key for downgrade compatibility - DO NOT DELETE
				console.log(
					`[Migration] localStorage: copied ${deprecatedKey} → ${newKey} (preserved old)`
				);
			}
		}
	}
};

// Migration 2: IndexedDB Database Name (Non-Destructive)

const IDXDB_MIGRATION_ID = 'idxdb-database-v1';

const idxdbMigration: Migration = {
	id: IDXDB_MIGRATION_ID,
	description: 'Copy IndexedDB from LlamacppWebui to LlamaUi database (non-destructive)',

	async run(): Promise<void> {
		const oldDbNames = await Dexie.getDatabaseNames();
		if (!oldDbNames.includes(DB_APP_NAME_DEPRECATED)) {
			console.log('[Migration] IndexedDB: no old database found, skipping');
			return;
		}

		// Check if new database already has data
		const newDb = new Dexie(STORAGE_APP_NAME);
		newDb.version(1).stores(IDXDB_STORES);
		const existingConvs = await newDb.table(IDXDB_TABLES.conversations).count();
		if (existingConvs > 0) {
			console.log('[Migration] IndexedDB: new database already has data, skipping');
			return;
		}

		console.log('[Migration] IndexedDB: copying from', DB_APP_NAME_DEPRECATED);

		const oldDb = new Dexie(DB_APP_NAME_DEPRECATED);
		oldDb.version(1).stores(IDXDB_STORES);

		const conversations = await oldDb.table(IDXDB_TABLES.conversations).toArray();
		const messages = await oldDb.table(IDXDB_TABLES.messages).toArray();

		if (conversations.length > 0) {
			await newDb.table(IDXDB_TABLES.conversations).bulkAdd(conversations);
			console.log(`[Migration] IndexedDB: copied ${conversations.length} conversations`);
		}
		if (messages.length > 0) {
			await newDb.table(IDXDB_TABLES.messages).bulkAdd(messages);
			console.log(`[Migration] IndexedDB: copied ${messages.length} messages`);
		}

		// Non-destructive: DO NOT delete old database - keep for downgrade compatibility
		console.log('[Migration] IndexedDB: preserved old database for downgrade compatibility');
	}
};

// Migration 3: Legacy Message Format

const LEGACY_MESSAGE_MIGRATION_ID = 'legacy-message-format-v2';

interface ParsedTurn {
	textBefore: string;
	toolCalls: Array<{ name: string; args: string; result: string }>;
}

function parseLegacyToolCalls(content: string): ParsedTurn[] {
	const turns: ParsedTurn[] = [];
	const regex = new RegExp(LEGACY_AGENTIC_REGEX.COMPLETED_TOOL_CALL.source, 'g');

	let lastIndex = 0;
	let currentTurn: ParsedTurn = { textBefore: '', toolCalls: [] };
	let match;

	while ((match = regex.exec(content)) !== null) {
		const textBefore = content.slice(lastIndex, match.index).trim();

		if (textBefore && currentTurn.toolCalls.length > 0) {
			turns.push(currentTurn);
			currentTurn = { textBefore, toolCalls: [] };
		} else if (textBefore && currentTurn.toolCalls.length === 0) {
			currentTurn.textBefore = textBefore;
		}

		currentTurn.toolCalls.push({
			name: match[1],
			args: match[2],
			result: match[3].replace(/^\n+|\n+$/g, '')
		});

		lastIndex = match.index + match[0].length;
	}

	const remainingText = content.slice(lastIndex).trim();

	if (currentTurn.toolCalls.length > 0) {
		turns.push(currentTurn);
	}

	if (remainingText) {
		const cleanRemaining = remainingText
			.replace(LEGACY_AGENTIC_REGEX.AGENTIC_TOOL_CALL_OPEN, '')
			.trim();
		if (cleanRemaining) {
			turns.push({ textBefore: cleanRemaining, toolCalls: [] });
		}
	}

	if (turns.length === 0) {
		turns.push({ textBefore: content.trim(), toolCalls: [] });
	}

	return turns;
}

function extractLegacyReasoning(content: string): { reasoning: string; cleanContent: string } {
	let reasoning = '';
	let cleanContent = content;

	const re = new RegExp(LEGACY_AGENTIC_REGEX.REASONING_EXTRACT.source, 'g');
	let match;
	while ((match = re.exec(content)) !== null) {
		reasoning += match[1];
	}

	cleanContent = cleanContent
		.replace(new RegExp(LEGACY_AGENTIC_REGEX.REASONING_BLOCK.source, 'g'), '')
		.replace(LEGACY_AGENTIC_REGEX.REASONING_OPEN, '');

	return { reasoning, cleanContent };
}

function hasLegacyMarkers(content: string): boolean {
	return LEGACY_AGENTIC_REGEX.HAS_LEGACY_MARKERS.test(content);
}

let DatabaseService: typeof import('./database.service').DatabaseService | null = null;

async function getDatabaseService() {
	if (!DatabaseService) {
		const module = await import('./database.service');
		DatabaseService = module.DatabaseService;
	}
	return DatabaseService;
}

const legacyMessageMigration: Migration = {
	id: LEGACY_MESSAGE_MIGRATION_ID,
	description: 'Migrate legacy marker-based messages to structured format',

	async run(): Promise<void> {
		const db = await getDatabaseService();
		const conversations = await db.getAllConversations();
		let migratedCount = 0;

		for (const conv of conversations) {
			const allMessages = await db.getConversationMessages(conv.id);

			for (const message of allMessages) {
				if (message.role !== MessageRole.ASSISTANT) {
					if (message.content?.includes(LEGACY_REASONING_TAGS.START)) {
						const { reasoning, cleanContent } = extractLegacyReasoning(message.content);
						await db.updateMessage(message.id, {
							content: cleanContent.trim(),
							reasoningContent: reasoning || undefined
						});
						migratedCount++;
					}
					continue;
				}

				if (!hasLegacyMarkers(message.content ?? '')) continue;

				const { reasoning, cleanContent } = extractLegacyReasoning(message.content);
				const turns = parseLegacyToolCalls(cleanContent);

				let existingToolCalls: Array<{
					id: string;
					function?: { name: string; arguments: string };
				}> = [];
				if (message.toolCalls) {
					try {
						existingToolCalls = JSON.parse(message.toolCalls);
					} catch {
						// Ignore
					}
				}

				const firstTurn = turns[0];
				if (!firstTurn) continue;

				const firstTurnToolCalls = firstTurn.toolCalls.map((tc, i) => {
					const existing =
						existingToolCalls.find((e) => e.function?.name === tc.name) || existingToolCalls[i];
					return {
						id: existing?.id || `legacy_tool_${i}`,
						type: 'function' as const,
						function: { name: tc.name, arguments: tc.args }
					};
				});

				await db.updateMessage(message.id, {
					content: firstTurn.textBefore,
					reasoningContent: reasoning || undefined,
					toolCalls: firstTurnToolCalls.length > 0 ? JSON.stringify(firstTurnToolCalls) : ''
				});

				let currentParentId = message.id;
				let toolCallIdCounter = existingToolCalls.length;

				for (let i = 0; i < firstTurn.toolCalls.length; i++) {
					const tc = firstTurn.toolCalls[i];
					const toolCallId = firstTurnToolCalls[i]?.id || `legacy_tool_${i}`;

					const toolMsg = await db.createMessageBranch(
						{
							convId: conv.id,
							type: 'text',
							role: MessageRole.TOOL,
							content: tc.result,
							toolCallId,
							timestamp: message.timestamp + i + 1,
							toolCalls: '',
							children: []
						},
						currentParentId
					);
					currentParentId = toolMsg.id;
				}

				for (let turnIdx = 1; turnIdx < turns.length; turnIdx++) {
					const turn = turns[turnIdx];

					const turnToolCalls = turn.toolCalls.map((tc, i) => {
						const idx = toolCallIdCounter + i;
						const existing = existingToolCalls[idx];
						return {
							id: existing?.id || `legacy_tool_${idx}`,
							type: 'function' as const,
							function: { name: tc.name, arguments: tc.args }
						};
					});
					toolCallIdCounter += turn.toolCalls.length;

					const assistantMsg = await db.createMessageBranch(
						{
							convId: conv.id,
							type: 'text',
							role: MessageRole.ASSISTANT,
							content: turn.textBefore,
							timestamp: message.timestamp + turnIdx * 100,
							toolCalls: turnToolCalls.length > 0 ? JSON.stringify(turnToolCalls) : '',
							children: [],
							model: message.model
						},
						currentParentId
					);
					currentParentId = assistantMsg.id;

					for (let i = 0; i < turn.toolCalls.length; i++) {
						const tc = turn.toolCalls[i];
						const toolCallId = turnToolCalls[i]?.id || `legacy_tool_${toolCallIdCounter + i}`;

						const toolMsg = await db.createMessageBranch(
							{
								convId: conv.id,
								type: 'text',
								role: MessageRole.TOOL,
								content: tc.result,
								toolCallId,
								timestamp: message.timestamp + turnIdx * 100 + i + 1,
								toolCalls: '',
								children: []
							},
							currentParentId
						);
						currentParentId = toolMsg.id;
					}
				}

				if (message.children.length > 0 && currentParentId !== message.id) {
					for (const childId of message.children) {
						const child = allMessages.find((m) => m.id === childId);
						if (!child) continue;
						if (child.role !== MessageRole.TOOL) {
							await db.updateMessage(childId, { parent: currentParentId });
						}
					}
					await db.updateMessage(message.id, { children: [] });
				}

				migratedCount++;
			}
		}

		console.log(`[Migration] Legacy messages: migrated ${migratedCount} messages`);
	}
};

// Migration 4: Theme Key (Non-Destructive)

const THEME_MIGRATION_ID = 'theme-key-v1';

const themeMigration: Migration = {
	id: THEME_MIGRATION_ID,
	description: 'Copy standalone theme key to config object (non-destructive)',

	async run(): Promise<void> {
		const legacyTheme = localStorage.getItem('theme');
		if (legacyTheme === null) {
			console.log('[Migration] Theme: no legacy theme key found, skipping');
			return;
		}

		// Check if config already has theme
		const configRaw = localStorage.getItem(CONFIG_LOCALSTORAGE_KEY);
		const config = configRaw ? JSON.parse(configRaw) : {};

		if (SETTINGS_KEYS.THEME in config) {
			console.log('[Migration] Theme: config already has theme, skipping');
			return;
		}

		config[SETTINGS_KEYS.THEME] = legacyTheme;
		localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(config));

		// Non-destructive: DO NOT delete legacy theme key - keep for downgrade compatibility
		console.log(`[Migration] Theme: copied standalone theme to config (preserved old key)`);
	}
};

// Migration Registry & Runner

const migrations: Migration[] = [
	localStorageMigration,
	idxdbMigration,
	legacyMessageMigration,
	themeMigration
];

export const MigrationService = {
	/**
	 * Get all registered migrations
	 */
	getMigrations(): Migration[] {
		return [...migrations];
	},

	/**
	 * Check if a specific migration has been completed
	 */
	isCompleted(id: string): boolean {
		return isMigrationCompleted(id);
	},

	/**
	 * Get current migration state
	 */
	getState(): MigrationState {
		return getMigrationState();
	},

	/**
	 * Reset migration state (use with caution - migrations will run again)
	 */
	resetState(): void {
		localStorage.removeItem(MIGRATION_STATE_KEY);
		console.log('[Migration] State reset - all migrations will run again');
	},

	/**
	 * Run all pending migrations (non-destructive - preserves legacy data)
	 * Should be called once at app initialization
	 */
	async runAllMigrations(): Promise<void> {
		const state = getMigrationState();
		console.log('[Migration] Starting migration run, state:', state);

		for (const migration of migrations) {
			if (isMigrationCompleted(migration.id)) {
				console.log(`[Migration] ${migration.id}: already completed, skipping`);
				continue;
			}

			try {
				console.log(`[Migration] ${migration.id}: running...`);
				await migration.run();
				markMigrationCompleted(migration.id);
				console.log(`[Migration] ${migration.id}: completed successfully`);
			} catch (error) {
				console.error(`[Migration] ${migration.id}: failed`, error);
				markMigrationFailed(migration.id);
			}
		}

		console.log('[Migration] All migrations complete');
	}
};
