import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { toast } from 'svelte-sonner';
import { DatabaseService } from '$lib/services/database';
import { config } from '$lib/stores/settings.svelte';
import { filterByLeafNodeId, findLeafNode } from '$lib/utils';
import { AttachmentType } from '$lib/enums';

/**
 * conversationsStore - Persistent conversation data and lifecycle management
 *
 * **Terminology - Chat vs Conversation:**
 * - **Chat**: The active interaction space with the Chat Completions API. Represents the
 *   real-time streaming session, loading states, and UI visualization of AI communication.
 *   Managed by chatStore, a "chat" is ephemeral and exists during active AI interactions.
 * - **Conversation**: The persistent database entity storing all messages and metadata.
 *   A "conversation" survives across sessions, page reloads, and browser restarts.
 *   It contains the complete message history, branching structure, and conversation metadata.
 *
 * This store manages all conversation-level data and operations including creation, loading,
 * deletion, and navigation. It maintains the list of conversations and the currently active
 * conversation with its message history, providing reactive state for UI components.
 *
 * **Architecture & Relationships:**
 * - **conversationsStore** (this class): Persistent conversation data management
 *   - Manages conversation list and active conversation state
 *   - Handles conversation CRUD operations via DatabaseService
 *   - Maintains active message array for current conversation
 *   - Coordinates branching navigation (currNode tracking)
 *
 * - **chatStore**: Uses conversation data as context for active AI streaming
 * - **DatabaseService**: Low-level IndexedDB storage for conversations and messages
 *
 * **Key Features:**
 * - **Conversation Lifecycle**: Create, load, update, delete conversations
 * - **Message Management**: Active message array with branching support
 * - **Import/Export**: JSON-based conversation backup and restore
 * - **Branch Navigation**: Navigate between message tree branches
 * - **Title Management**: Auto-update titles with confirmation dialogs
 * - **Reactive State**: Svelte 5 runes for automatic UI updates
 *
 * **State Properties:**
 * - `conversations`: All conversations sorted by last modified
 * - `activeConversation`: Currently viewed conversation
 * - `activeMessages`: Messages in current conversation path
 * - `isInitialized`: Store initialization status
 */
class ConversationsStore {
	// ─────────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────────

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

	// ─────────────────────────────────────────────────────────────────────────────
	// Modalities
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Modalities used in the active conversation.
	 * Computed from attachments in activeMessages.
	 * Used to filter available models - models must support all used modalities.
	 */
	usedModalities: ModelModalities = $derived.by(() => {
		return this.calculateModalitiesFromMessages(this.activeMessages);
	});

	/**
	 * Calculate modalities from a list of messages.
	 * Helper method used by both usedModalities and getModalitiesUpToMessage.
	 */
	private calculateModalitiesFromMessages(messages: DatabaseMessage[]): ModelModalities {
		const modalities: ModelModalities = { vision: false, audio: false };

		for (const message of messages) {
			if (!message.extra) continue;

			for (const extra of message.extra) {
				if (extra.type === AttachmentType.IMAGE) {
					modalities.vision = true;
				}

				// PDF only requires vision if processed as images
				if (extra.type === AttachmentType.PDF) {
					const pdfExtra = extra as DatabaseMessageExtraPdfFile;

					if (pdfExtra.processedAsImages) {
						modalities.vision = true;
					}
				}

				if (extra.type === AttachmentType.AUDIO) {
					modalities.audio = true;
				}
			}

			if (modalities.vision && modalities.audio) break;
		}

		return modalities;
	}

	/**
	 * Get modalities used in messages BEFORE the specified message.
	 * Used for regeneration - only consider context that was available when generating this message.
	 */
	getModalitiesUpToMessage(messageId: string): ModelModalities {
		const messageIndex = this.activeMessages.findIndex((m) => m.id === messageId);

		if (messageIndex === -1) {
			return this.usedModalities;
		}

		const messagesBefore = this.activeMessages.slice(0, messageIndex);
		return this.calculateModalitiesFromMessages(messagesBefore);
	}

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Lifecycle
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Initializes the conversations store by loading conversations from the database
	 */
	async initialize(): Promise<void> {
		try {
			await this.loadConversations();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize conversations store:', error);
		}
	}

	/**
	 * Loads all conversations from the database
	 */
	async loadConversations(): Promise<void> {
		this.conversations = await DatabaseService.getAllConversations();
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Conversation CRUD
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Creates a new conversation and navigates to it
	 * @param name - Optional name for the conversation
	 * @returns The ID of the created conversation
	 */
	async createConversation(name?: string): Promise<string> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;
		const conversation = await DatabaseService.createConversation(conversationName);

		this.conversations.unshift(conversation);
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
				this.activeMessages = filterByLeafNodeId(
					allMessages,
					conversation.currNode,
					false
				) as DatabaseMessage[];
			} else {
				this.activeMessages = await DatabaseService.getConversationMessages(convId);
			}

			return true;
		} catch (error) {
			console.error('Failed to load conversation:', error);
			return false;
		}
	}

	/**
	 * Clears the active conversation and messages
	 * Used when navigating away from chat or starting fresh
	 */
	clearActiveConversation(): void {
		this.activeConversation = null;
		this.activeMessages = [];
		// Active processing conversation is now managed by chatStore
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Message Management
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Refreshes active messages based on currNode after branch navigation
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

		this.activeMessages = [...currentPath];
	}

	/**
	 * Updates the name of a conversation
	 * @param convId - The conversation ID to update
	 * @param name - The new name for the conversation
	 */
	async updateConversationName(convId: string, name: string): Promise<void> {
		try {
			await DatabaseService.updateConversation(convId, { name });

			const convIndex = this.conversations.findIndex((c) => c.id === convId);

			if (convIndex !== -1) {
				this.conversations[convIndex].name = name;
			}

			if (this.activeConversation?.id === convId) {
				this.activeConversation.name = name;
			}
		} catch (error) {
			console.error('Failed to update conversation name:', error);
		}
	}

	/**
	 * Updates conversation title with optional confirmation dialog based on settings
	 * @param convId - The conversation ID to update
	 * @param newTitle - The new title content
	 * @param onConfirmationNeeded - Callback when user confirmation is needed
	 * @returns True if title was updated, false if cancelled
	 */
	async updateConversationTitleWithConfirmation(
		convId: string,
		newTitle: string,
		onConfirmationNeeded?: (currentTitle: string, newTitle: string) => Promise<boolean>
	): Promise<boolean> {
		try {
			const currentConfig = config();

			if (currentConfig.askForTitleConfirmation && onConfirmationNeeded) {
				const conversation = await DatabaseService.getConversation(convId);
				if (!conversation) return false;

				const shouldUpdate = await onConfirmationNeeded(conversation.name, newTitle);
				if (!shouldUpdate) return false;
			}

			await this.updateConversationName(convId, newTitle);
			return true;
		} catch (error) {
			console.error('Failed to update conversation title with confirmation:', error);
			return false;
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Navigation
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Updates the current node of the active conversation
	 * @param nodeId - The new current node ID
	 */
	async updateCurrentNode(nodeId: string): Promise<void> {
		if (!this.activeConversation) return;

		await DatabaseService.updateCurrentNode(this.activeConversation.id, nodeId);
		this.activeConversation.currNode = nodeId;
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
			this.conversations.unshift(updatedConv);
		}
	}

	/**
	 * Navigates to a specific sibling branch by updating currNode and refreshing messages
	 * @param siblingId - The sibling message ID to navigate to
	 */
	async navigateToSibling(siblingId: string): Promise<void> {
		if (!this.activeConversation) return;

		const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
		const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
		const currentFirstUserMessage = this.activeMessages.find(
			(m) => m.role === 'user' && m.parent === rootMessage?.id
		);

		const currentLeafNodeId = findLeafNode(allMessages, siblingId);

		await DatabaseService.updateCurrentNode(this.activeConversation.id, currentLeafNodeId);
		this.activeConversation.currNode = currentLeafNodeId;
		await this.refreshActiveMessages();

		// Only show title dialog if we're navigating between different first user message siblings
		if (rootMessage && this.activeMessages.length > 0) {
			const newFirstUserMessage = this.activeMessages.find(
				(m) => m.role === 'user' && m.parent === rootMessage.id
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
					newFirstUserMessage.content.trim(),
					this.titleUpdateConfirmationCallback
				);
			}
		}
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

	// ─────────────────────────────────────────────────────────────────────────────
	// Import/Export
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Downloads a conversation as JSON file
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
	 * Gets all messages for a specific conversation
	 * @param convId - The conversation ID
	 * @returns Array of messages
	 */
	async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await DatabaseService.getConversationMessages(convId);
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
	 * Adds a message to the active messages array
	 * Used by chatStore when creating new messages
	 * @param message - The message to add
	 */
	addMessageToActive(message: DatabaseMessage): void {
		this.activeMessages = [...this.activeMessages, message];
	}

	/**
	 * Updates a message at a specific index in active messages
	 * Creates a new object to trigger Svelte 5 reactivity
	 * @param index - The index of the message to update
	 * @param updates - Partial message data to update
	 */
	updateMessageAtIndex(index: number, updates: Partial<DatabaseMessage>): void {
		if (index !== -1 && this.activeMessages[index]) {
			// Create new object to trigger Svelte 5 reactivity
			const updated = { ...this.activeMessages[index], ...updates };
			this.activeMessages = [
				...this.activeMessages.slice(0, index),
				updated,
				...this.activeMessages.slice(index + 1)
			];
		}
	}

	/**
	 * Finds the index of a message in active messages
	 * @param messageId - The message ID to find
	 * @returns The index of the message, or -1 if not found
	 */
	findMessageIndex(messageId: string): number {
		return this.activeMessages.findIndex((m) => m.id === messageId);
	}

	/**
	 * Removes messages from active messages starting at an index
	 * @param startIndex - The index to start removing from
	 */
	sliceActiveMessages(startIndex: number): void {
		this.activeMessages = this.activeMessages.slice(0, startIndex);
	}

	/**
	 * Removes a message from active messages by index
	 * @param index - The index to remove
	 * @returns The removed message or undefined
	 */
	removeMessageAtIndex(index: number): DatabaseMessage | undefined {
		if (index !== -1) {
			const removed = this.activeMessages[index];
			this.activeMessages = [
				...this.activeMessages.slice(0, index),
				...this.activeMessages.slice(index + 1)
			];
			return removed;
		}
		return undefined;
	}

	/**
	 * Triggers file download in browser
	 * @param data - The data to download
	 * @param filename - Optional filename for the download
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

	// ─────────────────────────────────────────────────────────────────────────────
	// Utilities
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Sets the callback function for title update confirmations
	 * @param callback - Function to call when confirmation is needed
	 */
	setTitleUpdateConfirmationCallback(
		callback: (currentTitle: string, newTitle: string) => Promise<boolean>
	): void {
		this.titleUpdateConfirmationCallback = callback;
	}
}

export const conversationsStore = new ConversationsStore();

export const conversations = () => conversationsStore.conversations;
export const activeConversation = () => conversationsStore.activeConversation;
export const activeMessages = () => conversationsStore.activeMessages;
export const isConversationsInitialized = () => conversationsStore.isInitialized;
export const usedModalities = () => conversationsStore.usedModalities;
