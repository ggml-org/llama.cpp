import Dexie, { type EntityTable } from 'dexie';
import { findDescendantMessages } from '$lib/utils';

class LlamacppDatabase extends Dexie {
	conversations!: EntityTable<DatabaseConversation, string>;
	messages!: EntityTable<DatabaseMessage, string>;

	constructor() {
		super('LlamacppWebui');

		this.version(1).stores({
			conversations: 'id, lastModified, currNode, name',
			messages: 'id, convId, type, role, timestamp, parent, children'
		});
	}
}

const db = new LlamacppDatabase();
import { v4 as uuid } from 'uuid';
import { MessageRole } from '$lib/enums';

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

		await db.conversations.add(conversation);
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
		return await db.transaction('rw', [db.conversations, db.messages], async () => {
			// Handle null parent (root message case)
			if (parentId !== null) {
				const parentMessage = await db.messages.get(parentId);
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

			await db.messages.add(newMessage);

			// Update parent's children array if parent exists
			if (parentId !== null) {
				const parentMessage = await db.messages.get(parentId);
				if (parentMessage) {
					await db.messages.update(parentId, {
						children: [...parentMessage.children, newMessage.id]
					});
				}
			}

			await this.updateConversation(message.convId, {
				currNode: newMessage.id
			});

			return newMessage;
		});
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

		await db.messages.add(rootMessage);
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

		await db.messages.add(systemMessage);

		const parentMessage = await db.messages.get(parentId);
		if (parentMessage) {
			await db.messages.update(parentId, {
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
	static async deleteConversation(id: string): Promise<void> {
		await db.transaction('rw', [db.conversations, db.messages], async () => {
			await db.conversations.delete(id);
			await db.messages.where('convId').equals(id).delete();
		});
	}

	/**
	 * Deletes a message and removes it from its parent's children array.
	 *
	 * @param messageId - ID of the message to delete
	 */
	static async deleteMessage(messageId: string): Promise<void> {
		await db.transaction('rw', db.messages, async () => {
			const message = await db.messages.get(messageId);
			if (!message) return;

			// Remove this message from its parent's children array
			if (message.parent) {
				const parent = await db.messages.get(message.parent);
				if (parent) {
					parent.children = parent.children.filter((childId: string) => childId !== messageId);
					await db.messages.put(parent);
				}
			}

			// Delete the message
			await db.messages.delete(messageId);
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
		return await db.transaction('rw', db.messages, async () => {
			// Get all messages in the conversation to find descendants
			const allMessages = await db.messages.where('convId').equals(conversationId).toArray();

			// Find all descendant messages
			const descendants = findDescendantMessages(allMessages, messageId);
			const allToDelete = [messageId, ...descendants];

			// Get the message to delete for parent cleanup
			const message = await db.messages.get(messageId);
			if (message && message.parent) {
				const parent = await db.messages.get(message.parent);
				if (parent) {
					parent.children = parent.children.filter((childId: string) => childId !== messageId);
					await db.messages.put(parent);
				}
			}

			// Delete all messages in the branch
			await db.messages.bulkDelete(allToDelete);

			return allToDelete;
		});
	}

	/**
	 * Gets all conversations, sorted by last modified time (newest first).
	 *
	 * @returns Array of conversations
	 */
	static async getAllConversations(): Promise<DatabaseConversation[]> {
		return await db.conversations.orderBy('lastModified').reverse().toArray();
	}

	/**
	 * Gets a conversation by ID.
	 *
	 * @param id - Conversation ID
	 * @returns The conversation if found, otherwise undefined
	 */
	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		return await db.conversations.get(id);
	}

	/**
	 * Gets all messages in a conversation, sorted by timestamp (oldest first).
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the conversation
	 */
	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await db.messages.where('convId').equals(convId).sortBy('timestamp');
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
		await db.conversations.update(id, {
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
		await db.messages.update(id, updates);
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

		return await db.transaction('rw', [db.conversations, db.messages], async () => {
			for (const item of data) {
				const { conv, messages } = item;

				const existing = await db.conversations.get(conv.id);
				if (existing) {
					console.warn(`Conversation "${conv.name}" already exists, skipping...`);
					skippedCount++;
					continue;
				}

				await db.conversations.add(conv);
				for (const msg of messages) {
					await db.messages.put(msg);
				}

				importedCount++;
			}

			return { imported: importedCount, skipped: skippedCount };
		});
	}
}
