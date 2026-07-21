import Dexie, { type EntityTable } from 'dexie';
import { findDescendantMessages, uuid, filterByLeafNodeId } from '$lib/utils';
import { IDXDB_TABLES, IDXDB_STORES, STORAGE_APP_NAME } from '$lib/constants';
import { MessageRole } from '$lib/enums';
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
	 * @returns The created conversation
	 */
	static async createConversation(name: string): Promise<DatabaseConversation> {
		const conversation: DatabaseConversation = {
			id: uuid(),
			name,
			lastModified: Date.now(),
			currNode: ''
		};

		await db[IDXDB_TABLES.conversations].add(conversation);
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

				const newMessage: DatabaseMessage = {
					...message,
					id: uuid(),
					parent: parentId,
					toolCalls: message.toolCalls ?? '',
					children: []
				};

				await db[IDXDB_TABLES.messages].add(newMessage);

				// Update parent's children array if parent exists
				if (parentId !== null) {
					const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
					if (parentMessage) {
						await db[IDXDB_TABLES.messages].update(parentId, {
							children: [...parentMessage.children, newMessage.id]
						});
					}
				}

				await this.updateConversation(message.convId, {
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

		await db[IDXDB_TABLES.messages].add(rootMessage);
		return rootMessage.id;
	}

	/**
	 * Creates a system prompt message for a conversation.
	 *
	 * @param convId - Conversation ID
	 * @param systemPrompt - The system prompt content (must be non-empty)
	 * @param parentId - Parent message ID (typically the root message)
	 * @returns The created system message
	 * @throws Error if systemPrompt is empty
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

		await db[IDXDB_TABLES.messages].add(systemMessage);

		const parentMessage = await db[IDXDB_TABLES.messages].get(parentId);
		if (parentMessage) {
			await db[IDXDB_TABLES.messages].update(parentId, {
				children: [...parentMessage.children, systemMessage.id]
			});
		}

		return systemMessage;
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
		return await db[IDXDB_TABLES.conversations].orderBy('lastModified').reverse().toArray();
	}

	/**
	 * Gets a conversation by ID.
	 *
	 * @param id - Conversation ID
	 * @returns The conversation if found, otherwise undefined
	 */
	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		return await db[IDXDB_TABLES.conversations].get(id);
	}

	/**
	 * Gets all messages in a conversation, sorted by timestamp (oldest first).
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the conversation
	 */
	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await db[IDXDB_TABLES.messages].where('convId').equals(convId).sortBy('timestamp');
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
			db[IDXDB_TABLES.conversations].bulkGet(cleanIds),
			db[IDXDB_TABLES.messages].where('convId').anyOf(cleanIds).toArray()
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
	 * Updates a conversation.
	 *
	 * @param id - Conversation ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the conversation is updated
	 */
	static async updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): Promise<void> {
		await db[IDXDB_TABLES.conversations].update(id, {
			...updates,
			lastModified: Date.now()
		});
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
		const conversation = await db.conversations.get(id);
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

		const now = Date.now();
		await db.transaction('rw', db[IDXDB_TABLES.conversations], async () => {
			const convs = await db[IDXDB_TABLES.conversations].bulkGet(cleanIds);
			const updates: DatabaseConversation[] = [];
			for (let i = 0; i < cleanIds.length; i++) {
				const conv = convs[i];
				if (!conv) continue;
				const newPinned = !conv.pinned;
				updates.push({ ...conv, pinned: newPinned, lastModified: now });
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
		await db[IDXDB_TABLES.messages].update(id, updates);
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
	 */
	static async importConversations(
		data: { conv: DatabaseConversation; messages: DatabaseMessage[] }[]
	): Promise<{ imported: number; skipped: number }> {
		let importedCount = 0;
		let skippedCount = 0;

		return await db.transaction(
			'rw',
			[db[IDXDB_TABLES.conversations], db[IDXDB_TABLES.messages]],
			async () => {
				for (const item of data) {
					const { conv, messages } = item;

					const existing = await db[IDXDB_TABLES.conversations].get(conv.id);
					if (existing) {
						console.warn(`Conversation "${conv.name}" already exists, skipping...`);
						skippedCount++;
						continue;
					}

					await db[IDXDB_TABLES.conversations].add(conv);
					for (const msg of messages) {
						await db[IDXDB_TABLES.messages].put(msg);
					}

					importedCount++;
				}

				return { imported: importedCount, skipped: skippedCount };
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
					name: options.name,
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

				return newConv;
			}
		);
	}
}
