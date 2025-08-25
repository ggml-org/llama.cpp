import { db } from '$lib/stores/database';
import { filterByLeafNodeId, findDescendantMessages } from '$lib/utils/branching';

export class DatabaseService {
	static async createConversation(name: string): Promise<DatabaseConversation> {
		const conversation: DatabaseConversation = {
			id: crypto.randomUUID(),
			name,
			lastModified: Date.now(),
			currNode: ''
		};

		await db.conversations.add(conversation);
		return conversation;
	}

	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		return await db.conversations.get(id);
	}

	static async getAllConversations(): Promise<DatabaseConversation[]> {
		return await db.conversations.orderBy('lastModified').reverse().toArray();
	}

	static async updateConversation(id: string, updates: Partial<Omit<DatabaseConversation, 'id'>>): Promise<void> {
		await db.conversations.update(id, {
			...updates,
			lastModified: Date.now()
		});
	}

	static async deleteConversation(id: string): Promise<void> {
		await db.transaction('rw', [db.conversations, db.messages], async () => {
			await db.conversations.delete(id);
			await db.messages.where('convId').equals(id).delete();
		});
	}

	static async addMessage(
		message: Omit<DatabaseMessage, 'id'>
	): Promise<DatabaseMessage> {
		const newMessage: DatabaseMessage = {
			...message,
			id: crypto.randomUUID()
		};

		await db.messages.add(newMessage);
		return newMessage;
	}

	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await db.messages.where('convId').equals(convId).sortBy('timestamp');
	}

	static async updateMessage(
		id: string,
		updates: Partial<Omit<DatabaseMessage, 'id'>>
	): Promise<void> {
		await db.messages.update(id, updates);
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
	static async deleteMessageCascading(conversationId: string, messageId: string): Promise<string[]> {
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

	static async getSettings(): Promise<DatabaseAppSettings> {
		let settings = await db.settings.get('default');

		if (!settings) {
			settings = {
				id: 'default',
				theme: 'system',
				model: 'llama-3.2-3b-instruct',
				temperature: 0.7,
				maxTokens: 2048,
				topP: 0.9,
				topK: 40,
				repeatPenalty: 1.1,
				seed: -1,
				systemPrompt: 'You are a helpful AI assistant.'
			};
			await db.settings.add(settings);
		}

		return settings;
	}

	static async updateSettings(updates: Partial<Omit<DatabaseAppSettings, 'id'>>): Promise<void> {
		await db.settings.update('default', updates);
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
		const leafNodeId = conversation.currNode || 
			allMessages.reduce((latest, msg) => 
				msg.timestamp > latest.timestamp ? msg : latest
			).id;

		return filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];
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
				id: crypto.randomUUID(),
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
	 * Creates a root message for a new conversation.
	 * Root messages are not displayed but serve as the tree root for branching.
	 * 
	 * @param convId - Conversation ID
	 * @returns The created root message
	 */
	static async createRootMessage(convId: string): Promise<string> {
		const rootMessage: DatabaseMessage = {
			id: crypto.randomUUID(),
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
	 * Gets all leaf nodes (messages with no children) in a conversation.
	 * Useful for finding all possible conversation endpoints.
	 * 
	 * @param convId - Conversation ID
	 * @returns Array of leaf node message IDs
	 */
	static async getConversationLeafNodes(convId: string): Promise<string[]> {
		const allMessages = await this.getConversationMessages(convId);
		return allMessages
			.filter(msg => msg.children.length === 0)
			.map(msg => msg.id);
	}
}
