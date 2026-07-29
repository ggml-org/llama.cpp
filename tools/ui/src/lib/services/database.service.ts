import Dexie, { type EntityTable } from 'dexie';
import { findDescendantMessages, uuid, filterByLeafNodeId } from '$lib/utils';
import { IDXDB_TABLES, IDXDB_STORES, STORAGE_APP_NAME } from '$lib/constants';
import { AttachmentType, MessageRole } from '$lib/enums';
import { EncryptionService } from './encryption.service';
import type { McpServerOverride } from '$lib/types/database';
import type { ExportedConversation } from '$lib/types/database';

class LlamaUiDatabase extends Dexie {
	[IDXDB_TABLES.conversations]!: EntityTable<DatabaseConversation, string>;
	[IDXDB_TABLES.messages]!: EntityTable<DatabaseMessage, string>;

	constructor() {
		super(STORAGE_APP_NAME);

		this.version(1).stores(IDXDB_STORES);
	}
}

const db = new LlamaUiDatabase();

// Encryption seam: the fields below are encrypted at rest when encryption is
// enabled and unlocked. Values already carrying the encrypted prefix are
// never re-encrypted, and non-prefixed values pass through on read, so
// records written before encryption was enabled keep working.
//
// Transforms must run OUTSIDE Dexie transactions: awaiting WebCrypto inside a
// transaction would commit it prematurely.

const MESSAGE_SECRET_FIELDS = ['content', 'reasoningContent', 'toolCalls'] as const;

function assertEncryptionWritable(): void {
	if (EncryptionService.isEnabled() && !EncryptionService.isUnlocked()) {
		throw new Error('Encryption is locked');
	}
}

async function encryptName(name: string): Promise<string> {
	if (!EncryptionService.isUnlocked() || name === '' || EncryptionService.isEncryptedValue(name)) {
		return name;
	}
	return await EncryptionService.encryptString(name);
}

async function encryptConversationSecrets<T extends Partial<DatabaseConversation>>(
	record: T
): Promise<T> {
	if (typeof record.name !== 'string') return record;
	const name = await encryptName(record.name);
	return name === record.name ? record : { ...record, name };
}

async function decryptConversationSecrets(
	conversation: DatabaseConversation
): Promise<DatabaseConversation> {
	if (!EncryptionService.isUnlocked() || !EncryptionService.isEncryptedValue(conversation.name)) {
		return conversation;
	}
	try {
		return { ...conversation, name: await EncryptionService.decryptString(conversation.name) };
	} catch {
		return conversation;
	}
}

async function encryptMessageSecrets<T extends Partial<DatabaseMessage>>(record: T): Promise<T> {
	if (!EncryptionService.isUnlocked()) return record;

	let changed = false;
	const result = { ...record };
	for (const field of MESSAGE_SECRET_FIELDS) {
		const value = result[field];
		if (typeof value === 'string' && value !== '' && !EncryptionService.isEncryptedValue(value)) {
			(result as Record<string, unknown>)[field] = await EncryptionService.encryptString(value);
			changed = true;
		}
	}

	if (Array.isArray(result.extra) && result.extra.length > 0) {
		const marker = result.extra[0] as unknown as DatabaseMessageExtraEncrypted;
		const alreadyEncrypted = result.extra.length === 1 && marker.type === AttachmentType.ENCRYPTED;
		if (!alreadyEncrypted) {
			const next: DatabaseMessageExtraEncrypted = {
				type: AttachmentType.ENCRYPTED,
				payload: await EncryptionService.encryptJson(result.extra)
			};
			result.extra = [next as unknown as DatabaseMessageExtra];
			changed = true;
		}
	}
	return changed ? result : record;
}

async function decryptMessageSecrets(message: DatabaseMessage): Promise<DatabaseMessage> {
	if (!EncryptionService.isUnlocked()) return message;

	let changed = false;
	const result = { ...message };
	for (const field of MESSAGE_SECRET_FIELDS) {
		const value = result[field];
		if (EncryptionService.isEncryptedValue(value)) {
			try {
				(result as Record<string, unknown>)[field] = await EncryptionService.decryptString(value);
				changed = true;
			} catch {
				// literal text starting with the prefix, or corrupt data: keep as stored
			}
		}
	}

	const firstExtra = result.extra?.[0] as unknown as DatabaseMessageExtraEncrypted | undefined;
	if (result.extra?.length === 1 && firstExtra?.type === AttachmentType.ENCRYPTED) {
		try {
			result.extra = await EncryptionService.decryptJson<DatabaseMessageExtra[]>(
				firstExtra.payload
			);
			changed = true;
		} catch {
			// keep as stored
		}
	}
	return changed ? result : message;
}

export class DatabaseService {
	/**
	 *
	 *
	 * Conversations
	 *
	 *
	 */

	/**
	 * Creates a new conversation.
	 *
	 * @param name - Name of the conversation
	 * @param fields - Optional extra fields (e.g. reasoningEffort)
	 * @returns The created conversation
	 */
	static async createConversation(
		name: string,
		fields?: Partial<Omit<DatabaseConversation, 'id' | 'name' | 'lastModified'>>
	): Promise<DatabaseConversation> {
		assertEncryptionWritable();

		const conversation: DatabaseConversation = {
			id: uuid(),
			name,
			lastModified: Date.now(),
			currNode: '',
			...fields
		};

		await db[IDXDB_TABLES.conversations].add(await encryptConversationSecrets(conversation));
		return conversation;
	}

	/**
	 *
	 *
	 * Messages
	 *
	 *
	 */

	/**
	 * Creates a new message branch by adding a message and updating parent/child relationships.
	 * Also updates the conversation's currNode to point to the new message.
	 *
	 * @param message - Message to add (without id)
	 * @param parentId - Parent message ID to attach to
	 * @returns The created message
	 */
	static async createMessageBranch(
		message: Omit<DatabaseMessage, 'id'>,
		parentId: string | null
	): Promise<DatabaseMessage> {
		assertEncryptionWritable();

		const newMessage: DatabaseMessage = {
			...message,
			id: uuid(),
			parent: parentId,
			toolCalls: message.toolCalls ?? '',
			children: []
		};
		// Encrypt before entering the transaction: awaiting WebCrypto inside a
		// Dexie transaction would commit it prematurely.
		const stored = await encryptMessageSecrets(newMessage);

		return await db.transaction(
			'rw',
			[db[IDXDB_TABLES.conversations], db[IDXDB_TABLES.messages]],
			async () => {
				// Handle null parent (root message case)
				if (parentId !== null) {
					const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
					if (!parentMessage) {
						throw new Error(`Parent message ${parentId} not found`);
					}
				}

				await db[IDXDB_TABLES.messages].add(stored);

				// Update parent's children array if parent exists
				if (parentId !== null) {
					const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
					if (parentMessage) {
						await db[IDXDB_TABLES.messages].update(parentId, {
							children: [...parentMessage.children, newMessage.id]
						});
					}
				}

				await db[IDXDB_TABLES.conversations].update(message.convId, {
					currNode: newMessage.id
				});

				return newMessage;
			}
		);
	}

	/**
	 * Creates a root message for a new conversation.
	 * Root messages are not displayed but serve as the tree root for branching.
	 *
	 * @param convId - Conversation ID
	 * @returns The created root message
	 */
	static async createRootMessage(convId: string): Promise<string> {
		assertEncryptionWritable();

		const rootMessage: DatabaseMessage = {
			id: uuid(),
			convId,
			type: 'root',
			timestamp: Date.now(),
			role: MessageRole.SYSTEM,
			content: '',
			parent: null,
			toolCalls: '',
			children: []
		};

		await db[IDXDB_TABLES.messages].add(await encryptMessageSecrets(rootMessage));
		return rootMessage.id;
	}

	/**
	 * Creates a system prompt message for a conversation.
	 *
	 * @param convId - Conversation ID
	 * @param systemPrompt - The system prompt content (must be non-empty)
	 * @param parentId - Parent message ID (typically the root message)
	 * @returns The created system message
	 * @throws Error if systemPrompt is empty or the parent message does not exist
	 */
	static async createSystemMessage(
		convId: string,
		systemPrompt: string,
		parentId: string
	): Promise<DatabaseMessage> {
		const trimmedPrompt = systemPrompt.trim();
		if (!trimmedPrompt) {
			throw new Error('Cannot create system message with empty content');
		}

		assertEncryptionWritable();

		const systemMessage: DatabaseMessage = {
			id: uuid(),
			convId,
			type: MessageRole.SYSTEM,
			timestamp: Date.now(),
			role: MessageRole.SYSTEM,
			content: trimmedPrompt,
			parent: parentId,
			children: []
		};
		const stored = await encryptMessageSecrets(systemMessage);

		return await db.transaction('rw', db[IDXDB_TABLES.messages], async () => {
			const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
			if (!parentMessage) {
				throw new Error(`Parent message ${parentId} not found`);
			}

			await db[IDXDB_TABLES.messages].add(stored);
			await db[IDXDB_TABLES.messages].update(parentId, {
				children: [...parentMessage.children, systemMessage.id]
			});

			return systemMessage;
		});
	}

	/**
	 * Deletes a conversation and all its messages.
	 *
	 * @param id - Conversation ID
	 */
	static async deleteConversation(
		id: string,
		options?: { deleteWithForks?: boolean }
	): Promise<void> {
		await db.transaction(
			'rw',
			[db[IDXDB_TABLES.conversations], db[IDXDB_TABLES.messages]],
			async () => {
				if (options?.deleteWithForks) {
					// Recursively collect all descendant IDs
					const idsToDelete: string[] = [];
					const queue = [id];

					while (queue.length > 0) {
						const parentId = queue.pop()!;
						const children = await db[IDXDB_TABLES.conversations]
							.filter((c) => c.forkedFromConversationId === parentId)
							.toArray();

						for (const child of children) {
							idsToDelete.push(child.id);
							queue.push(child.id);
						}
					}

					for (const forkId of idsToDelete) {
						await db[IDXDB_TABLES.conversations].delete(forkId);
						await db[IDXDB_TABLES.messages].where('convId').equals(forkId).delete();
					}
				} else {
					await this.reparentDirectChildren(id);
				}

				await db[IDXDB_TABLES.conversations].delete(id);
				await db[IDXDB_TABLES.messages].where('convId').equals(id).delete();
			}
		);
	}

	/**
	 * Reparents direct children of `parentId` to the nearest surviving
	 * ancestor (or promotes them to top-level when the immediate parent was
	 * top-level). Walking skips any ancestor listed in `excludeIds`, since
	 * those will be deleted in the same batch — leaving a grandchild pointing
	 * at an `excludeIds` entry would orphan it. Children whose own id is in
	 * `excludeIds` are dropped from the updates (the bulk-delete pass will
	 * remove them). `prefetched` may carry a pre-fetched ancestor map to
	 * avoid repeat reads inside a bulk transaction.
	 */
	private static async reparentDirectChildren(
		parentId: string,
		excludeIds: ReadonlySet<string> = new Set(),
		prefetched?: ReadonlyMap<string, DatabaseConversation>
	): Promise<void> {
		const conv = prefetched?.get(parentId) ?? (await db[IDXDB_TABLES.conversations].get(parentId));
		if (!conv) return;

		let newParent = conv.forkedFromConversationId;
		const visited = new Set<string>([parentId]);
		while (newParent && excludeIds.has(newParent)) {
			if (visited.has(newParent)) {
				newParent = undefined;
				break;
			}
			visited.add(newParent);
			const next =
				prefetched?.get(newParent) ?? (await db[IDXDB_TABLES.conversations].get(newParent));
			if (!next) {
				newParent = undefined;
				break;
			}
			newParent = next.forkedFromConversationId;
		}

		const directChildren = await db[IDXDB_TABLES.conversations]
			.filter((c) => c.forkedFromConversationId === parentId)
			.toArray();

		const updates: DatabaseConversation[] = [];
		for (const child of directChildren) {
			if (excludeIds.has(child.id)) continue;
			updates.push({ ...child, forkedFromConversationId: newParent });
		}
		if (updates.length === 0) return;
		await db[IDXDB_TABLES.conversations].bulkPut(updates);
	}

	/**
	 * Deletes multiple conversations in a single transaction. Each deleted
	 * conversation has its direct children reparented to the nearest surviving
	 * ancestor (or promoted to top-level). Children also in `ids` are dropped
	 * entirely rather than reparented.
	 *
	 * @param ids - Conversation IDs to delete
	 */
	static async bulkDeleteConversations(ids: string[]): Promise<void> {
		const cleanIds = ids.filter((id): id is string => typeof id === 'string' && id.length > 0);
		if (cleanIds.length === 0) return;
		const idSet = new Set(cleanIds);

		await db.transaction(
			'rw',
			[db[IDXDB_TABLES.conversations], db[IDXDB_TABLES.messages]],
			async () => {
				// Pre-load each to-delete conversation so the per-id reparent
				// walk-up doesn't ping-pong the same ancestry chain.
				const prefetched = new Map<string, DatabaseConversation>();
				let frontier = [...cleanIds];
				const requested = new Set<string>(frontier);
				while (frontier.length > 0) {
					const fetched = await db[IDXDB_TABLES.conversations].bulkGet(frontier);
					frontier = [];
					for (let i = 0; i < fetched.length; i++) {
						const conv = fetched[i];
						if (!conv || !conv.id) continue;
						prefetched.set(conv.id, conv);
						const ancestor = conv.forkedFromConversationId;
						if (ancestor && !prefetched.has(ancestor) && !requested.has(ancestor)) {
							frontier.push(ancestor);
							requested.add(ancestor);
						}
					}
				}

				for (const id of cleanIds) {
					await this.reparentDirectChildren(id, idSet, prefetched);
				}

				await db[IDXDB_TABLES.conversations].bulkDelete(cleanIds);
				await db[IDXDB_TABLES.messages].where('convId').anyOf(cleanIds).delete();
			}
		);
	}

	/**
	 * Deletes a message and removes it from its parent's children array.
	 *
	 * @param messageId - ID of the message to delete
	 */
	static async deleteMessage(messageId: string): Promise<void> {
		await db.transaction('rw', db[IDXDB_TABLES.messages], async () => {
			const message = await db[IDXDB_TABLES.messages].get(messageId);
			if (!message) return;

			// Remove this message from its parent's children array
			if (message.parent) {
				const parent = await db[IDXDB_TABLES.messages].get(message.parent);
				if (parent) {
					parent.children = parent.children.filter((childId: string) => childId !== messageId);
					await db[IDXDB_TABLES.messages].put(parent);
				}
			}

			// Delete the message
			await db[IDXDB_TABLES.messages].delete(messageId);
		});
	}

	/**
	 * Deletes a message and all its descendant messages (cascading deletion).
	 * This removes the entire branch starting from the specified message.
	 *
	 * @param conversationId - ID of the conversation containing the message
	 * @param messageId - ID of the root message to delete (along with all descendants)
	 * @returns Array of all deleted message IDs
	 */
	static async deleteMessageCascading(
		conversationId: string,
		messageId: string
	): Promise<string[]> {
		return await db.transaction('rw', db[IDXDB_TABLES.messages], async () => {
			// Get all messages in the conversation to find descendants
			const allMessages = await db[IDXDB_TABLES.messages]
				.where('convId')
				.equals(conversationId)
				.toArray();

			// Find all descendant messages
			const descendants = findDescendantMessages(allMessages, messageId);
			const allToDelete = [messageId, ...descendants];

			// Get the message to delete for parent cleanup
			const message = await db[IDXDB_TABLES.messages].get(messageId);
			if (message && message.parent) {
				const parent = await db[IDXDB_TABLES.messages].get(message.parent);
				if (parent) {
					parent.children = parent.children.filter((childId: string) => childId !== messageId);
					await db[IDXDB_TABLES.messages].put(parent);
				}
			}

			// Delete all messages in the branch
			await db[IDXDB_TABLES.messages].bulkDelete(allToDelete);

			return allToDelete;
		});
	}

	/**
	 * Gets all conversations, sorted by last modified time (newest first).
	 *
	 * @returns Array of conversations
	 */
	static async getAllConversations(): Promise<DatabaseConversation[]> {
		const conversations = await db[IDXDB_TABLES.conversations]
			.orderBy('lastModified')
			.reverse()
			.toArray();
		return await Promise.all(conversations.map(decryptConversationSecrets));
	}

	/**
	 * Gets a conversation by ID.
	 *
	 * @param id - Conversation ID
	 * @returns The conversation if found, otherwise undefined
	 */
	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		const conversation = await db[IDXDB_TABLES.conversations].get(id);
		return conversation ? await decryptConversationSecrets(conversation) : undefined;
	}

	/**
	 * Gets all messages in a conversation, sorted by timestamp (oldest first).
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the conversation
	 */
	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		const messages = await db[IDXDB_TABLES.messages]
			.where('convId')
			.equals(convId)
			.sortBy('timestamp');
		return await Promise.all(messages.map(decryptMessageSecrets));
	}

	/**
	 * Loads multiple conversations with all of their messages in two bulk
	 * reads. Missing conversations are silently omitted from the result.
	 *
	 * @param convIds - Conversation IDs to load
	 * @returns Map of id -> { conv, messages }. Messages are sorted ascending by timestamp.
	 */
	static async getConversationsWithMessages(
		convIds: string[]
	): Promise<Map<string, ExportedConversation>> {
		const result = new Map<string, ExportedConversation>();
		const cleanIds = convIds.filter((id): id is string => typeof id === 'string' && id.length > 0);
		if (cleanIds.length === 0) return result;

		const [convs, allMessages] = await Promise.all([
			db[IDXDB_TABLES.conversations]
				.bulkGet(cleanIds)
				.then((items) =>
					Promise.all(items.map((conv) => (conv ? decryptConversationSecrets(conv) : conv)))
				),
			db[IDXDB_TABLES.messages]
				.where('convId')
				.anyOf(cleanIds)
				.toArray()
				.then((messages) => Promise.all(messages.map(decryptMessageSecrets)))
		]);

		const messagesByConv = new Map<string, DatabaseMessage[]>();
		for (const msg of allMessages) {
			const bucket = messagesByConv.get(msg.convId);
			if (bucket) bucket.push(msg);
			else messagesByConv.set(msg.convId, [msg]);
		}

		for (let i = 0; i < cleanIds.length; i++) {
			const conv = convs[i];
			if (!conv) continue;
			const messages = (messagesByConv.get(conv.id) ?? []).sort(
				(a, b) => a.timestamp - b.timestamp
			);
			result.set(conv.id, { conv, messages });
		}
		return result;
	}

	/**
	 * Updates a conversation. `lastModified` is never stamped implicitly;
	 * pass it in `updates` to bump the conversation in recency ordering.
	 *
	 * @param id - Conversation ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the conversation is updated
	 */
	static async updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): Promise<void> {
		assertEncryptionWritable();
		await db[IDXDB_TABLES.conversations].update(id, await encryptConversationSecrets(updates));
	}

	/**
	 *
	 *
	 * Navigation
	 *
	 *
	 */

	/**
	 * Toggles the pinned status of a conversation.
	 *
	 * @param id - Conversation ID
	 * @returns The new pinned status
	 */
	static async toggleConversationPin(id: string): Promise<boolean> {
		const conversation = await db[IDXDB_TABLES.conversations].get(id);
		if (!conversation) {
			throw new Error(`Conversation ${id} not found`);
		}
		const newPinnedState = !conversation.pinned;
		await this.updateConversation(id, { pinned: newPinnedState });
		return newPinnedState;
	}

	/**
	 * Toggles the pinned status of each conversation in `ids` inside a single
	 * transaction. Treats `pinned === undefined` as `false`, matching the
	 * semantics of {@link toggleConversationPin} where `!undefined` evaluates
	 * to `true`. Returns the resulting pinned state for every id that was
	 * updated; missing ids are omitted from the map.
	 *
	 * @param ids - Conversation IDs to toggle
	 * @returns Map of id -> new pinned state
	 */
	static async bulkToggleConversationPins(ids: string[]): Promise<Map<string, boolean>> {
		const cleanIds = ids.filter((id): id is string => typeof id === 'string' && id.length > 0);
		const result = new Map<string, boolean>();
		if (cleanIds.length === 0) return result;

		await db.transaction('rw', db[IDXDB_TABLES.conversations], async () => {
			const convs = await db[IDXDB_TABLES.conversations].bulkGet(cleanIds);
			const updates: DatabaseConversation[] = [];
			for (let i = 0; i < cleanIds.length; i++) {
				const conv = convs[i];
				if (!conv) continue;
				const newPinned = !conv.pinned;
				updates.push({ ...conv, pinned: newPinned });
				result.set(cleanIds[i], newPinned);
			}
			if (updates.length === 0) return;
			await db[IDXDB_TABLES.conversations].bulkPut(updates);
		});
		return result;
	}

	/**
	 * Updates the conversation's current node (active branch).
	 * This determines which conversation path is currently being viewed.
	 *
	 * @param convId - Conversation ID
	 * @param nodeId - Message ID to set as current node
	 */
	static async updateCurrentNode(convId: string, nodeId: string): Promise<void> {
		await this.updateConversation(convId, {
			currNode: nodeId
		});
	}

	/**
	 * Updates a message.
	 *
	 * @param id - Message ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the message is updated
	 */
	static async updateMessage(
		id: string,
		updates: Partial<Omit<DatabaseMessage, 'id'>>
	): Promise<void> {
		assertEncryptionWritable();
		await db[IDXDB_TABLES.messages].update(id, await encryptMessageSecrets(updates));
	}

	/**
	 * Re-encrypts every stored conversation and message with the session key.
	 * Records already encrypted are left untouched, so the pass is idempotent
	 * and can simply be re-run after an interruption. Requires the unlocked
	 * session key; run when encryption is enabled.
	 */
	static async encryptAllStoredData(): Promise<void> {
		if (!EncryptionService.isUnlocked()) {
			throw new Error('Encryption is locked');
		}

		const conversations = await db[IDXDB_TABLES.conversations].toArray();
		const reEncryptedConversations = await Promise.all(
			conversations.map(async (conv) => {
				const next = await encryptConversationSecrets(conv);
				return next === conv ? null : next;
			})
		);
		await db[IDXDB_TABLES.conversations].bulkPut(
			reEncryptedConversations.filter(
				(conv: DatabaseConversation | null): conv is DatabaseConversation => conv !== null
			)
		);

		const messages = await db[IDXDB_TABLES.messages].toArray();
		const reEncryptedMessages = await Promise.all(
			messages.map(async (message) => {
				const next = await encryptMessageSecrets(message);
				return next === message ? null : next;
			})
		);
		await db[IDXDB_TABLES.messages].bulkPut(
			reEncryptedMessages.filter(
				(message: DatabaseMessage | null): message is DatabaseMessage => message !== null
			)
		);
	}

	/**
	 * Encrypts already-fetched conversations for an encrypted export. Values
	 * already carrying the encrypted prefix pass through untouched.
	 *
	 * @param data - Plaintext exported conversations
	 * @returns Copies with all secret fields encrypted
	 */
	static async encryptForExport(data: ExportedConversation[]): Promise<ExportedConversation[]> {
		if (!EncryptionService.isUnlocked()) {
			throw new Error('Encryption is locked');
		}
		return await Promise.all(
			data.map(async ({ conv, messages }) => ({
				conv: await encryptConversationSecrets(conv),
				messages: await Promise.all(messages.map((msg) => encryptMessageSecrets(msg)))
			}))
		);
	}

	/**
	 * Decrypts every stored conversation and message back to plaintext.
	 * Records already plaintext are left untouched, so the pass is idempotent
	 * and can simply be re-run after an interruption. Requires the unlocked
	 * session key; run before encryption is disabled.
	 */
	static async decryptAllStoredData(): Promise<void> {
		if (!EncryptionService.isUnlocked()) {
			throw new Error('Encryption is locked');
		}

		const conversations = await db[IDXDB_TABLES.conversations].toArray();
		const decryptedConversations = await Promise.all(
			conversations.map(async (conv) => {
				const next = await decryptConversationSecrets(conv);
				return next === conv ? null : next;
			})
		);
		await db[IDXDB_TABLES.conversations].bulkPut(
			decryptedConversations.filter(
				(conv: DatabaseConversation | null): conv is DatabaseConversation => conv !== null
			)
		);

		const messages = await db[IDXDB_TABLES.messages].toArray();
		const decryptedMessages = await Promise.all(
			messages.map(async (message) => {
				const next = await decryptMessageSecrets(message);
				return next === message ? null : next;
			})
		);
		await db[IDXDB_TABLES.messages].bulkPut(
			decryptedMessages.filter(
				(message: DatabaseMessage | null): message is DatabaseMessage => message !== null
			)
		);
	}

	/**
	 *
	 *
	 * Import
	 *
	 *
	 */

	/**
	 * Imports multiple conversations and their messages.
	 * Skips conversations that already exist.
	 *
	 * @param data - Array of { conv, messages } objects
	 * @returns The conversations written to the database and the ones skipped
	 */
	static async importConversations(
		data: { conv: DatabaseConversation; messages: DatabaseMessage[] }[]
	): Promise<{ imported: DatabaseConversation[]; skipped: DatabaseConversation[] }> {
		assertEncryptionWritable();
		const imported: DatabaseConversation[] = [];
		const skipped: DatabaseConversation[] = [];

		// Encrypt before entering the transaction: awaiting WebCrypto inside a
		// Dexie transaction would commit it prematurely.
		const prepared = await Promise.all(
			data.map(async (item) => ({
				original: item.conv,
				conv: await encryptConversationSecrets(item.conv),
				messages: await Promise.all(item.messages.map((msg) => encryptMessageSecrets(msg)))
			}))
		);

		return await db.transaction(
			'rw',
			[db[IDXDB_TABLES.conversations], db[IDXDB_TABLES.messages]],
			async () => {
				for (const item of prepared) {
					const { conv, messages, original } = item;

					const existing = await db[IDXDB_TABLES.conversations].get(conv.id);
					if (existing) {
						skipped.push(original);
						continue;
					}

					await db[IDXDB_TABLES.conversations].add(conv);
					for (const msg of messages) {
						await db[IDXDB_TABLES.messages].put(msg);
					}

					imported.push(original);
				}

				return { imported, skipped };
			}
		);
	}

	/**
	 *
	 *
	 * Forking
	 *
	 *
	 */

	/**
	 * Forks a conversation at a specific message, creating a new conversation
	 * containing all messages from the root up to (and including) the target message.
	 *
	 * @param sourceConvId - The source conversation ID
	 * @param atMessageId - The message ID to fork at (the new conversation ends here)
	 * @param options - Fork options (name and whether to include attachments)
	 * @returns The newly created conversation
	 */
	static async forkConversation(
		sourceConvId: string,
		atMessageId: string,
		options: { name: string; includeAttachments: boolean }
	): Promise<DatabaseConversation> {
		assertEncryptionWritable();
		const storedName = await encryptName(options.name);

		return await db.transaction(
			'rw',
			[db[IDXDB_TABLES.conversations], db[IDXDB_TABLES.messages]],
			async () => {
				const sourceConv = await db[IDXDB_TABLES.conversations].get(sourceConvId);
				if (!sourceConv) {
					throw new Error(`Source conversation ${sourceConvId} not found`);
				}

				const allMessages = await db[IDXDB_TABLES.messages]
					.where('convId')
					.equals(sourceConvId)
					.toArray();

				const pathMessages = filterByLeafNodeId(
					allMessages,
					atMessageId,
					true
				) as DatabaseMessage[];
				if (pathMessages.length === 0) {
					throw new Error(`Could not resolve message path to ${atMessageId}`);
				}

				const idMap = new Map<string, string>();

				for (const msg of pathMessages) {
					idMap.set(msg.id, uuid());
				}

				const newConvId = uuid();
				const clonedMessages: DatabaseMessage[] = pathMessages.map((msg) => {
					const newId = idMap.get(msg.id)!;
					const newParent = msg.parent ? (idMap.get(msg.parent) ?? null) : null;
					const newChildren = msg.children
						.filter((childId: string) => idMap.has(childId))
						.map((childId: string) => idMap.get(childId)!);

					return {
						...msg,
						id: newId,
						convId: newConvId,
						parent: newParent,
						children: newChildren,
						extra: options.includeAttachments ? msg.extra : undefined
					};
				});

				const lastClonedMessage = clonedMessages[clonedMessages.length - 1];
				const newConv: DatabaseConversation = {
					id: newConvId,
					name: storedName,
					lastModified: Date.now(),
					currNode: lastClonedMessage.id,
					forkedFromConversationId: sourceConvId,
					mcpServerOverrides: sourceConv.mcpServerOverrides
						? sourceConv.mcpServerOverrides.map((o: McpServerOverride) => ({
								serverId: o.serverId,
								enabled: o.enabled
							}))
						: undefined
				};

				await db[IDXDB_TABLES.conversations].add(newConv);

				for (const msg of clonedMessages) {
					await db[IDXDB_TABLES.messages].add(msg);
				}

				// Return the plaintext record; only the stored copy is encrypted
				return { ...newConv, name: options.name };
			}
		);
	}
}
