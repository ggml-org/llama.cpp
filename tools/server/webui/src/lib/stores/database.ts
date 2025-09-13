import Dexie, { type EntityTable } from 'dexie';
import { filterByLeafNodeId, findDescendantMessages } from '$lib/utils/branching';

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

/**
 * DatabaseStore - Persistent data layer for conversation and message management
 *
 * This service provides a comprehensive data access layer built on IndexedDB using Dexie.
 * It handles all persistent storage operations for conversations, messages, and application settings
 * with support for complex conversation branching and message threading.
 *
 * **Architecture & Relationships:**
 * - **DatabaseStore** (this class): Stateless data persistence layer
 *   - Manages IndexedDB operations through Dexie ORM
 *   - Handles conversation and message CRUD operations
 *   - Supports complex branching with parent-child relationships
 *   - Provides transaction safety for multi-table operations
 *
 * - **ChatStore**: Primary consumer for conversation state management
 *   - Uses DatabaseStore for all persistence operations
 *   - Coordinates UI state with database state
 *   - Handles conversation lifecycle and message branching
 *
 * **Key Features:**
 * - **Conversation Management**: Create, read, update, delete conversations
 * - **Message Branching**: Support for tree-like conversation structures
 * - **Transaction Safety**: Atomic operations for data consistency
 * - **Path Resolution**: Navigate conversation branches and find leaf nodes
 * - **Cascading Deletion**: Remove entire conversation branches
 *
 * **Database Schema:**
 * - `conversations`: Conversation metadata with current node tracking
 * - `messages`: Individual messages with parent-child relationships
 *
 * **Branching Model:**
 * Messages form a tree structure where each message can have multiple children,
 * enabling conversation branching and alternative response paths. The conversation's
 * `currNode` tracks the currently active branch endpoint.
 */
import { v4 as uuid } from 'uuid';

export class DatabaseStore {
	/**
	 * Adds a new message to the database.
	 *
	 * @param message - Message to add (without id)
	 * @returns The created message
	 */
	static async addMessage(message: Omit<DatabaseMessage, 'id'>): Promise<DatabaseMessage> {
		const newMessage: DatabaseMessage = {
			...message,
			id: uuid()
		};

		await db.messages.add(newMessage);
		return newMessage;
	}

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
			role: 'system',
			content: '',
			parent: null,
			thinking: '',
			children: []
		};

		await db.messages.add(rootMessage);
		return rootMessage.id;
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
	 * Gets all leaf nodes (messages with no children) in a conversation.
	 * Useful for finding all possible conversation endpoints.
	 *
	 * @param convId - Conversation ID
	 * @returns Array of leaf node message IDs
	 */
	static async getConversationLeafNodes(convId: string): Promise<string[]> {
		const allMessages = await this.getConversationMessages(convId);
		return allMessages.filter((msg) => msg.children.length === 0).map((msg) => msg.id);
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
	 * Gets the conversation path from root to the current leaf node.
	 * Uses the conversation's currNode to determine the active branch.
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the current conversation path
	 */
	static async getConversationPath(convId: string): Promise<DatabaseMessage[]> {
		const conversation = await this.getConversation(convId);

		if (!conversation) {
			return [];
		}

		const allMessages = await this.getConversationMessages(convId);

		if (allMessages.length === 0) {
			return [];
		}

		// If no currNode is set, use the latest message as leaf
		const leafNodeId =
			conversation.currNode ||
			allMessages.reduce((latest, msg) => (msg.timestamp > latest.timestamp ? msg : latest)).id;

		return filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];
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
}
