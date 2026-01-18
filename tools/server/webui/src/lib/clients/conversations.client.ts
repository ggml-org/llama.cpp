/**
 * ConversationsClient - Business Logic Facade for Conversation Operations
 *
 * Coordinates conversation lifecycle, persistence, and navigation.
 *
 * **Architecture & Relationships:**
 * - **ConversationsClient** (this class): Business logic facade
 *   - Uses DatabaseService for IndexedDB operations
 *   - Updates conversationsStore with reactive state
 *   - Handles CRUD, import/export, branch navigation
 *
 * - **DatabaseService**: Stateless IndexedDB layer
 * - **conversationsStore**: Reactive state only ($state)
 *
 * **Key Responsibilities:**
 * - Conversation lifecycle (create, load, delete)
 * - Message management and tree navigation
 * - MCP server per-chat overrides
 * - Import/Export functionality
 * - Title management with confirmation
 */

import { goto } from '$app/navigation';
import { toast } from 'svelte-sonner';
import { DatabaseService } from '$lib/services/database.service';
import { config } from '$lib/stores/settings.svelte';
import { filterByLeafNodeId, findLeafNode } from '$lib/utils';
import type { McpServerOverride } from '$lib/types/database';

interface ConversationsStoreStateCallbacks {
	getConversations: () => DatabaseConversation[];
	setConversations: (conversations: DatabaseConversation[]) => void;
	getActiveConversation: () => DatabaseConversation | null;
	setActiveConversation: (conversation: DatabaseConversation | null) => void;
	getActiveMessages: () => DatabaseMessage[];
	setActiveMessages: (messages: DatabaseMessage[]) => void;
	updateActiveMessages: (updater: (messages: DatabaseMessage[]) => DatabaseMessage[]) => void;
	setInitialized: (initialized: boolean) => void;
	getPendingMcpServerOverrides: () => McpServerOverride[];
	setPendingMcpServerOverrides: (overrides: McpServerOverride[]) => void;
	getTitleUpdateConfirmationCallback: () =>
		| ((currentTitle: string, newTitle: string) => Promise<boolean>)
		| undefined;
}

export class ConversationsClient {
	private storeCallbacks: ConversationsStoreStateCallbacks | null = null;

	/**
	 *
	 *
	 * Store Integration
	 *
	 *
	 */

	/**
	 * Sets callbacks for store state updates.
	 * Called by conversationsStore during initialization.
	 */
	setStoreCallbacks(callbacks: ConversationsStoreStateCallbacks): void {
		this.storeCallbacks = callbacks;
	}

	private get store(): ConversationsStoreStateCallbacks {
		if (!this.storeCallbacks) {
			throw new Error('ConversationsClient: Store callbacks not initialized');
		}
		return this.storeCallbacks;
	}

	/**
	 *
	 *
	 * Lifecycle
	 *
	 *
	 */

	/**
	 * Initializes the conversations by loading from the database.
	 */
	async initialize(): Promise<void> {
		try {
			await this.loadConversations();
			this.store.setInitialized(true);
		} catch (error) {
			console.error('Failed to initialize conversations:', error);
		}
	}

	/**
	 * Loads all conversations from the database
	 */
	async loadConversations(): Promise<void> {
		const conversations = await DatabaseService.getAllConversations();
		this.store.setConversations(conversations);
	}

	/**
	 * Creates a new conversation and navigates to it
	 * @param name - Optional name for the conversation
	 * @returns The ID of the created conversation
	 */
	async createConversation(name?: string): Promise<string> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;
		const conversation = await DatabaseService.createConversation(conversationName);

		const pendingOverrides = this.store.getPendingMcpServerOverrides();
		if (pendingOverrides.length > 0) {
			// Deep clone to plain objects (Svelte 5 $state uses Proxies which can't be cloned to IndexedDB)
			const plainOverrides = pendingOverrides.map((o) => ({
				serverId: o.serverId,
				enabled: o.enabled
			}));
			conversation.mcpServerOverrides = plainOverrides;
			await DatabaseService.updateConversation(conversation.id, {
				mcpServerOverrides: plainOverrides
			});
			this.store.setPendingMcpServerOverrides([]);
		}

		const conversations = this.store.getConversations();
		this.store.setConversations([conversation, ...conversations]);
		this.store.setActiveConversation(conversation);
		this.store.setActiveMessages([]);

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

			this.store.setPendingMcpServerOverrides([]);
			this.store.setActiveConversation(conversation);

			if (conversation.currNode) {
				const allMessages = await DatabaseService.getConversationMessages(convId);
				const filteredMessages = filterByLeafNodeId(
					allMessages,
					conversation.currNode,
					false
				) as DatabaseMessage[];
				this.store.setActiveMessages(filteredMessages);
			} else {
				const messages = await DatabaseService.getConversationMessages(convId);
				this.store.setActiveMessages(messages);
			}

			return true;
		} catch (error) {
			console.error('Failed to load conversation:', error);
			return false;
		}
	}

	/**
	 *
	 *
	 * Conversation CRUD
	 *
	 *
	 */

	/**
	 * Clears the active conversation and messages.
	 */
	clearActiveConversation(): void {
		this.store.setActiveConversation(null);
		this.store.setActiveMessages([]);
	}

	/**
	 * Deletes a conversation and all its messages
	 * @param convId - The conversation ID to delete
	 */
	async deleteConversation(convId: string): Promise<void> {
		try {
			await DatabaseService.deleteConversation(convId);

			const conversations = this.store.getConversations();
			this.store.setConversations(conversations.filter((c) => c.id !== convId));

			const activeConv = this.store.getActiveConversation();
			if (activeConv?.id === convId) {
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
			this.store.setConversations([]);

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
		const activeConv = this.store.getActiveConversation();
		if (!activeConv) return;

		const allMessages = await DatabaseService.getConversationMessages(activeConv.id);

		if (allMessages.length === 0) {
			this.store.setActiveMessages([]);
			return;
		}

		const leafNodeId =
			activeConv.currNode ||
			allMessages.reduce((latest, msg) => (msg.timestamp > latest.timestamp ? msg : latest)).id;

		const currentPath = filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];

		this.store.setActiveMessages(currentPath);
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

			const conversations = this.store.getConversations();
			const convIndex = conversations.findIndex((c) => c.id === convId);

			if (convIndex !== -1) {
				conversations[convIndex].name = name;
				this.store.setConversations([...conversations]);
			}

			const activeConv = this.store.getActiveConversation();
			if (activeConv?.id === convId) {
				this.store.setActiveConversation({ ...activeConv, name });
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
			const onConfirmationNeeded = this.store.getTitleUpdateConfirmationCallback();

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

	/**
	 * Updates conversation lastModified timestamp and moves it to top of list
	 */
	updateConversationTimestamp(): void {
		const activeConv = this.store.getActiveConversation();
		if (!activeConv) return;

		const conversations = this.store.getConversations();
		const chatIndex = conversations.findIndex((c) => c.id === activeConv.id);

		if (chatIndex !== -1) {
			conversations[chatIndex].lastModified = Date.now();
			const updatedConv = conversations.splice(chatIndex, 1)[0];
			this.store.setConversations([updatedConv, ...conversations]);
		}
	}

	/**
	 * Updates the current node of the active conversation
	 * @param nodeId - The new current node ID
	 */
	async updateCurrentNode(nodeId: string): Promise<void> {
		const activeConv = this.store.getActiveConversation();
		if (!activeConv) return;

		await DatabaseService.updateCurrentNode(activeConv.id, nodeId);
		this.store.setActiveConversation({ ...activeConv, currNode: nodeId });
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
		const activeConv = this.store.getActiveConversation();
		if (!activeConv) return;

		const allMessages = await DatabaseService.getConversationMessages(activeConv.id);
		const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
		const activeMessages = this.store.getActiveMessages();
		const currentFirstUserMessage = activeMessages.find(
			(m) => m.role === 'user' && m.parent === rootMessage?.id
		);

		const currentLeafNodeId = findLeafNode(allMessages, siblingId);

		await DatabaseService.updateCurrentNode(activeConv.id, currentLeafNodeId);
		this.store.setActiveConversation({ ...activeConv, currNode: currentLeafNodeId });
		await this.refreshActiveMessages();

		const updatedActiveMessages = this.store.getActiveMessages();
		if (rootMessage && updatedActiveMessages.length > 0) {
			const newFirstUserMessage = updatedActiveMessages.find(
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
					activeConv.id,
					newFirstUserMessage.content.trim()
				);
			}
		}
	}

	/**
	 *
	 *
	 * MCP Server Overrides
	 *
	 *
	 */

	/**
	 * Gets MCP server override for a specific server in the active conversation.
	 * Falls back to pending overrides if no active conversation exists.
	 * @param serverId - The server ID to check
	 * @returns The override if set, undefined if using global setting
	 */
	getMcpServerOverride(serverId: string): McpServerOverride | undefined {
		const activeConv = this.store.getActiveConversation();
		if (activeConv) {
			return activeConv.mcpServerOverrides?.find((o: McpServerOverride) => o.serverId === serverId);
		}
		return this.store.getPendingMcpServerOverrides().find((o) => o.serverId === serverId);
	}

	/**
	 * Checks if an MCP server is enabled for the active conversation.
	 * Per-chat override takes precedence over global setting.
	 * @param serverId - The server ID to check
	 * @param globalEnabled - The global enabled state from settings
	 * @returns True if server is enabled for this conversation
	 */
	isMcpServerEnabledForChat(serverId: string, globalEnabled: boolean): boolean {
		const override = this.getMcpServerOverride(serverId);
		return override !== undefined ? override.enabled : globalEnabled;
	}

	/**
	 * Sets or removes MCP server override for the active conversation.
	 * If no conversation exists, stores as pending override.
	 * @param serverId - The server ID to override
	 * @param enabled - The enabled state, or undefined to remove override
	 */
	async setMcpServerOverride(serverId: string, enabled: boolean | undefined): Promise<void> {
		const activeConv = this.store.getActiveConversation();

		if (!activeConv) {
			this.setPendingMcpServerOverride(serverId, enabled);
			return;
		}

		// Clone to plain objects to avoid Proxy serialization issues with IndexedDB
		const currentOverrides = (activeConv.mcpServerOverrides || []).map((o: McpServerOverride) => ({
			serverId: o.serverId,
			enabled: o.enabled
		}));
		let newOverrides: McpServerOverride[];

		if (enabled === undefined) {
			newOverrides = currentOverrides.filter((o: McpServerOverride) => o.serverId !== serverId);
		} else {
			const existingIndex = currentOverrides.findIndex(
				(o: McpServerOverride) => o.serverId === serverId
			);
			if (existingIndex >= 0) {
				newOverrides = [...currentOverrides];
				newOverrides[existingIndex] = { serverId, enabled };
			} else {
				newOverrides = [...currentOverrides, { serverId, enabled }];
			}
		}

		await DatabaseService.updateConversation(activeConv.id, {
			mcpServerOverrides: newOverrides.length > 0 ? newOverrides : undefined
		});

		const updatedConv = {
			...activeConv,
			mcpServerOverrides: newOverrides.length > 0 ? newOverrides : undefined
		};
		this.store.setActiveConversation(updatedConv);

		const conversations = this.store.getConversations();
		const convIndex = conversations.findIndex((c) => c.id === activeConv.id);
		if (convIndex !== -1) {
			conversations[convIndex].mcpServerOverrides =
				newOverrides.length > 0 ? newOverrides : undefined;
			this.store.setConversations([...conversations]);
		}
	}

	/**
	 * Toggles MCP server enabled state for the active conversation.
	 * @param serverId - The server ID to toggle
	 * @param globalEnabled - The global enabled state from settings
	 */
	async toggleMcpServerForChat(serverId: string, globalEnabled: boolean): Promise<void> {
		const currentEnabled = this.isMcpServerEnabledForChat(serverId, globalEnabled);
		await this.setMcpServerOverride(serverId, !currentEnabled);
	}

	/**
	 * Resets MCP server to use global setting (removes per-chat override).
	 * @param serverId - The server ID to reset
	 */
	async resetMcpServerToGlobal(serverId: string): Promise<void> {
		await this.setMcpServerOverride(serverId, undefined);
	}

	/**
	 * Sets or removes a pending MCP server override (for new conversations).
	 */
	private setPendingMcpServerOverride(serverId: string, enabled: boolean | undefined): void {
		const pendingOverrides = this.store.getPendingMcpServerOverrides();

		if (enabled === undefined) {
			this.store.setPendingMcpServerOverrides(
				pendingOverrides.filter((o) => o.serverId !== serverId)
			);
		} else {
			const existingIndex = pendingOverrides.findIndex((o) => o.serverId === serverId);
			if (existingIndex >= 0) {
				const newOverrides = [...pendingOverrides];
				newOverrides[existingIndex] = { serverId, enabled };
				this.store.setPendingMcpServerOverrides(newOverrides);
			} else {
				this.store.setPendingMcpServerOverrides([...pendingOverrides, { serverId, enabled }]);
			}
		}
	}

	/**
	 * Clears all pending MCP server overrides.
	 */
	clearPendingMcpServerOverrides(): void {
		this.store.setPendingMcpServerOverrides([]);
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

		const activeConv = this.store.getActiveConversation();
		if (activeConv?.id === convId) {
			conversation = activeConv;
			messages = this.store.getActiveMessages();
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

export const conversationsClient = new ConversationsClient();
