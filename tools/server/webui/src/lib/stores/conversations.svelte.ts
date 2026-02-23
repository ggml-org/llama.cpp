/**
 * conversationsStore - Reactive State Store for Conversations
 *
 * Manages conversation lifecycle, persistence, navigation.
 *
 * **Architecture & Relationships:**
 * - **DatabaseService**: Stateless IndexedDB layer
 * - **conversationsStore** (this): Reactive state + business logic
 * - **chatStore**: Chat-specific state (streaming, loading)
 *
 * **Key Responsibilities:**
 * - Conversation CRUD (create, load, delete)
 * - Message management and tree navigation
 * - Import/Export functionality
 * - Title management with confirmation
 *
 * @see DatabaseService in services/database.ts for IndexedDB operations
 */

import { goto } from '$app/navigation';
import { browser } from '$app/environment';
import { toast } from 'svelte-sonner';
import { DatabaseService } from '$lib/services/database.service';
import { config } from '$lib/stores/settings.svelte';
import { filterByLeafNodeId, findLeafNode } from '$lib/utils';
import { MessageRole } from '$lib/enums';

class ConversationsStore {
	/**
	 *
	 *
	 * State
	 *
	 *
	 */

	/** List of all conversations */
	conversations = $state<DatabaseConversation[]>([]);

	/** Currently active conversation */
	activeConversation = $state<DatabaseConversation | null>(null);

	/** Messages in the active conversation (filtered by currNode path) */
	activeMessages = $state<DatabaseMessage[]>([]);

	/** Whether the store has been initialized */
	isInitialized = $state(false);

	/** Callback for title update confirmation dialog */
	titleUpdateConfirmationCallback?: (currentTitle: string, newTitle: string) => Promise<boolean>;

	/**
	 *
	 *
	 * Lifecycle
	 *
	 *
	 */

	/**
	 * Initialize the store by loading conversations from database.
	 * Must be called once after app startup.
	 */
	async init(): Promise<void> {
		if (!browser) return;
		if (this.isInitialized) return;

		try {
			await this.loadConversations();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize conversations:', error);
		}
	}

	/**
	 * Alias for init() for backward compatibility.
	 */
	async initialize(): Promise<void> {
		return this.init();
	}

	/**
	 *
	 *
	 * Message Array Operations
	 *
	 *
	 */

	/**
	 * Adds a message to the active messages array
	 */
	addMessageToActive(message: DatabaseMessage): void {
		this.activeMessages.push(message);
	}

	/**
	 * Updates a message at a specific index in active messages
	 */
	updateMessageAtIndex(index: number, updates: Partial<DatabaseMessage>): void {
		if (index !== -1 && this.activeMessages[index]) {
			this.activeMessages[index] = { ...this.activeMessages[index], ...updates };
		}
	}

	/**
	 * Finds the index of a message in active messages
	 */
	findMessageIndex(messageId: string): number {
		return this.activeMessages.findIndex((m) => m.id === messageId);
	}

	/**
	 * Removes messages from active messages starting at an index
	 */
	sliceActiveMessages(startIndex: number): void {
		this.activeMessages = this.activeMessages.slice(0, startIndex);
	}

	/**
	 * Removes a message from active messages by index
	 */
	removeMessageAtIndex(index: number): DatabaseMessage | undefined {
		if (index !== -1) {
			return this.activeMessages.splice(index, 1)[0];
		}
		return undefined;
	}

	/**
	 * Sets the callback function for title update confirmations
	 */
	setTitleUpdateConfirmationCallback(
		callback: (currentTitle: string, newTitle: string) => Promise<boolean>
	): void {
		this.titleUpdateConfirmationCallback = callback;
	}

	/**
	 *
	 *
	 * Conversation CRUD
	 *
	 *
	 */

	/**
	 * Loads all conversations from the database
	 */
	async loadConversations(): Promise<void> {
		const conversations = await DatabaseService.getAllConversations();
		this.conversations = conversations;
	}

	/**
	 * Creates a new conversation and navigates to it
	 * @param name - Optional name for the conversation
	 * @returns The ID of the created conversation
	 */
	async createConversation(name?: string): Promise<string> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;
		const conversation = await DatabaseService.createConversation(conversationName);

		this.conversations = [conversation, ...this.conversations];
		this.activeConversation = conversation;
		this.activeMessages = [];

		await goto(`#/chat/${conversation.id}`);

		return conversation.id;
	}

	/**
	 * Loads a specific conversation and its messages
	 * @param convId - The conversation ID to load
	 * @returns True if conversation was loaded successfully
	 */
	async loadConversation(convId: string): Promise<boolean> {
		try {
			const conversation = await DatabaseService.getConversation(convId);

			if (!conversation) {
				return false;
			}

			this.activeConversation = conversation;

			if (conversation.currNode) {
				const allMessages = await DatabaseService.getConversationMessages(convId);
				const filteredMessages = filterByLeafNodeId(
					allMessages,
					conversation.currNode,
					false
				) as DatabaseMessage[];
				this.activeMessages = filteredMessages;
			} else {
				const messages = await DatabaseService.getConversationMessages(convId);
				this.activeMessages = messages;
			}

			return true;
		} catch (error) {
			console.error('Failed to load conversation:', error);
			return false;
		}
	}

	/**
	 * Clears the active conversation and messages.
	 */
	clearActiveConversation(): void {
		this.activeConversation = null;
		this.activeMessages = [];
	}

	/**
	 * Deletes a conversation and all its messages
	 * @param convId - The conversation ID to delete
	 */
	async deleteConversation(convId: string): Promise<void> {
		try {
			await DatabaseService.deleteConversation(convId);

			this.conversations = this.conversations.filter((c) => c.id !== convId);

			if (this.activeConversation?.id === convId) {
				this.clearActiveConversation();
				await goto(`?new_chat=true#/`);
			}
		} catch (error) {
			console.error('Failed to delete conversation:', error);
		}
	}

	/**
	 * Deletes all conversations and their messages
	 */
	async deleteAll(): Promise<void> {
		try {
			const allConversations = await DatabaseService.getAllConversations();

			for (const conv of allConversations) {
				await DatabaseService.deleteConversation(conv.id);
			}

			this.clearActiveConversation();
			this.conversations = [];

			toast.success('All conversations deleted');

			await goto(`?new_chat=true#/`);
		} catch (error) {
			console.error('Failed to delete all conversations:', error);
			toast.error('Failed to delete conversations');
		}
	}

	/**
	 *
	 *
	 * Message Management
	 *
	 *
	 */

	/**
	 * Refreshes active messages based on currNode after branch navigation.
	 */
	async refreshActiveMessages(): Promise<void> {
		if (!this.activeConversation) return;

		const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);

		if (allMessages.length === 0) {
			this.activeMessages = [];
			return;
		}

		const leafNodeId =
			this.activeConversation.currNode ||
			allMessages.reduce((latest, msg) => (msg.timestamp > latest.timestamp ? msg : latest)).id;

		const currentPath = filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];

		this.activeMessages = currentPath;
	}

	/**
	 * Gets all messages for a specific conversation
	 * @param convId - The conversation ID
	 * @returns Array of messages
	 */
	async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await DatabaseService.getConversationMessages(convId);
	}

	/**
	 *
	 *
	 * Title Management
	 *
	 *
	 */

	/**
	 * Updates the name of a conversation.
	 * @param convId - The conversation ID to update
	 * @param name - The new name for the conversation
	 */
	async updateConversationName(convId: string, name: string): Promise<void> {
		try {
			await DatabaseService.updateConversation(convId, { name });

			const convIndex = this.conversations.findIndex((c) => c.id === convId);

			if (convIndex !== -1) {
				this.conversations[convIndex].name = name;
				this.conversations = [...this.conversations];
			}

			if (this.activeConversation?.id === convId) {
				this.activeConversation = { ...this.activeConversation, name };
			}
		} catch (error) {
			console.error('Failed to update conversation name:', error);
		}
	}

	/**
	 * Updates conversation title with optional confirmation dialog based on settings
	 * @param convId - The conversation ID to update
	 * @param newTitle - The new title content
	 * @returns True if title was updated, false if cancelled
	 */
	async updateConversationTitleWithConfirmation(
		convId: string,
		newTitle: string
	): Promise<boolean> {
		try {
			const currentConfig = config();

			if (currentConfig.askForTitleConfirmation && this.titleUpdateConfirmationCallback) {
				const conversation = await DatabaseService.getConversation(convId);
				if (!conversation) return false;

				const shouldUpdate = await this.titleUpdateConfirmationCallback(
					conversation.name,
					newTitle
				);
				if (!shouldUpdate) return false;
			}

			await this.updateConversationName(convId, newTitle);
			return true;
		} catch (error) {
			console.error('Failed to update conversation title with confirmation:', error);
			return false;
		}
	}

	/**
	 * Updates conversation lastModified timestamp and moves it to top of list
	 */
	updateConversationTimestamp(): void {
		if (!this.activeConversation) return;

		const chatIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);

		if (chatIndex !== -1) {
			this.conversations[chatIndex].lastModified = Date.now();
			const updatedConv = this.conversations.splice(chatIndex, 1)[0];
			this.conversations = [updatedConv, ...this.conversations];
		}
	}

	/**
	 * Updates the current node of the active conversation
	 * @param nodeId - The new current node ID
	 */
	async updateCurrentNode(nodeId: string): Promise<void> {
		if (!this.activeConversation) return;

		await DatabaseService.updateCurrentNode(this.activeConversation.id, nodeId);
		this.activeConversation = { ...this.activeConversation, currNode: nodeId };
	}

	/**
	 *
	 *
	 * Branch Navigation
	 *
	 *
	 */

	/**
	 * Navigates to a specific sibling branch by updating currNode and refreshing messages.
	 * @param siblingId - The sibling message ID to navigate to
	 */
	async navigateToSibling(siblingId: string): Promise<void> {
		if (!this.activeConversation) return;

		const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
		const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
		const currentFirstUserMessage = this.activeMessages.find(
			(m) => m.role === MessageRole.USER && m.parent === rootMessage?.id
		);

		const currentLeafNodeId = findLeafNode(allMessages, siblingId);

		await DatabaseService.updateCurrentNode(this.activeConversation.id, currentLeafNodeId);
		this.activeConversation = { ...this.activeConversation, currNode: currentLeafNodeId };
		await this.refreshActiveMessages();

		if (rootMessage && this.activeMessages.length > 0) {
			const newFirstUserMessage = this.activeMessages.find(
				(m) => m.role === MessageRole.USER && m.parent === rootMessage.id
			);

			if (
				newFirstUserMessage &&
				newFirstUserMessage.content.trim() &&
				(!currentFirstUserMessage ||
					newFirstUserMessage.id !== currentFirstUserMessage.id ||
					newFirstUserMessage.content.trim() !== currentFirstUserMessage.content.trim())
			) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newFirstUserMessage.content.trim()
				);
			}
		}
	}

	/**
	 *
	 *
	 * Import & Export
	 *
	 *
	 */

	/**
	 * Downloads a conversation as JSON file.
	 * @param convId - The conversation ID to download
	 */
	async downloadConversation(convId: string): Promise<void> {
		let conversation: DatabaseConversation | null;
		let messages: DatabaseMessage[];

		if (this.activeConversation?.id === convId) {
			conversation = this.activeConversation;
			messages = this.activeMessages;
		} else {
			conversation = await DatabaseService.getConversation(convId);
			if (!conversation) return;
			messages = await DatabaseService.getConversationMessages(convId);
		}

		this.triggerDownload({ conv: conversation, messages });
	}

	/**
	 * Exports all conversations with their messages as a JSON file
	 * @returns The list of exported conversations
	 */
	async exportAllConversations(): Promise<DatabaseConversation[]> {
		const allConversations = await DatabaseService.getAllConversations();

		if (allConversations.length === 0) {
			throw new Error('No conversations to export');
		}

		const allData = await Promise.all(
			allConversations.map(async (conv) => {
				const messages = await DatabaseService.getConversationMessages(conv.id);
				return { conv, messages };
			})
		);

		const blob = new Blob([JSON.stringify(allData, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `all_conversations_${new Date().toISOString().split('T')[0]}.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);

		toast.success(`All conversations (${allConversations.length}) prepared for download`);

		return allConversations;
	}

	/**
	 * Imports conversations from a JSON file
	 * Opens file picker and processes the selected file
	 * @returns The list of imported conversations
	 */
	async importConversations(): Promise<DatabaseConversation[]> {
		return new Promise((resolve, reject) => {
			const input = document.createElement('input');
			input.type = 'file';
			input.accept = '.json';

			input.onchange = async (e) => {
				const file = (e.target as HTMLInputElement)?.files?.[0];

				if (!file) {
					reject(new Error('No file selected'));
					return;
				}

				try {
					const text = await file.text();
					const parsedData = JSON.parse(text);
					let importedData: ExportedConversations;

					if (Array.isArray(parsedData)) {
						importedData = parsedData;
					} else if (
						parsedData &&
						typeof parsedData === 'object' &&
						'conv' in parsedData &&
						'messages' in parsedData
					) {
						importedData = [parsedData];
					} else {
						throw new Error('Invalid file format');
					}

					const result = await DatabaseService.importConversations(importedData);
					toast.success(`Imported ${result.imported} conversation(s), skipped ${result.skipped}`);

					await this.loadConversations();

					const importedConversations = (
						Array.isArray(importedData) ? importedData : [importedData]
					).map((item) => item.conv);

					resolve(importedConversations);
				} catch (err: unknown) {
					const message = err instanceof Error ? err.message : 'Unknown error';
					console.error('Failed to import conversations:', err);
					toast.error('Import failed', { description: message });
					reject(new Error(`Import failed: ${message}`));
				}
			};

			input.click();
		});
	}

	/**
	 * Imports conversations from provided data (without file picker)
	 * @param data - Array of conversation data with messages
	 * @returns Import result with counts
	 */
	async importConversationsData(
		data: ExportedConversations
	): Promise<{ imported: number; skipped: number }> {
		const result = await DatabaseService.importConversations(data);
		await this.loadConversations();
		return result;
	}

	/**
	 * Triggers file download in browser
	 */
	private triggerDownload(data: ExportedConversations, filename?: string): void {
		const conversation =
			'conv' in data ? data.conv : Array.isArray(data) ? data[0]?.conv : undefined;

		if (!conversation) {
			console.error('Invalid data: missing conversation');
			return;
		}

		const conversationName = conversation.name?.trim() || '';
		const truncatedSuffix = conversationName
			.toLowerCase()
			.replace(/[^a-z0-9]/gi, '_')
			.replace(/_+/g, '_')
			.substring(0, 20);
		const downloadFilename = filename || `conversation_${conversation.id}_${truncatedSuffix}.json`;

		const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = downloadFilename;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}
}

export const conversationsStore = new ConversationsStore();

// Auto-initialize in browser
if (browser) {
	conversationsStore.init();
}

export const conversations = () => conversationsStore.conversations;
export const activeConversation = () => conversationsStore.activeConversation;
export const activeMessages = () => conversationsStore.activeMessages;
export const isConversationsInitialized = () => conversationsStore.isInitialized;
