import Dexie, { type EntityTable } from 'dexie';
import { findDescendantMessages, uuid, filterByLeafNodeId } from '$lib/utils';
import { IDXDB_TABLES, IDXDB_STORES, STORAGE_APP_NAME } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import type { McpServerOverride } from '$lib/types/database';

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
					// Reparent direct children to deleted conv's parent
					const conv = await db[IDXDB_TABLES.conversations].get(id);
					const newParent = conv?.forkedFromConversationId;
					const directChildren = await db[IDXDB_TABLES.conversations]
						.filter((c) => c.forkedFromConversationId === id)
						.toArray();

					for (const child of directChildren) {
						await db[IDXDB_TABLES.conversations].update(child.id, {
							forkedFromConversationId: newParent ?? undefined
						});
					}
				}

				await db[IDXDB_TABLES.conversations].delete(id);
				await db[IDXDB_TABLES.messages].where('convId').equals(id).delete();
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
