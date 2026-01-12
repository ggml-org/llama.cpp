/**
 * conversationsStore - Reactive State Store for Conversations
 *
 * This store contains ONLY reactive state ($state, $derived).
 * All business logic is delegated to ConversationsClient.
 *
 * **Architecture & Relationships:**
 * - **ConversationsClient**: Business logic facade (CRUD, navigation, import/export)
 * - **DatabaseService**: Stateless IndexedDB layer
 * - **conversationsStore** (this): Reactive state for UI components
 *
 * **Responsibilities:**
 * - Hold reactive state for UI binding
 * - Provide getters for computed values
 * - Expose setters for ConversationsClient to update state
 * - Forward method calls to ConversationsClient
 *
 * @see ConversationsClient in clients/ for business logic
 * @see DatabaseService in services/database.ts for IndexedDB operations
 */

import { browser } from '$app/environment';
import { AttachmentType } from '$lib/enums';
import type { McpServerOverride } from '$lib/types/database';

class ConversationsStore {
	/** List of all conversations */
	conversations = $state<DatabaseConversation[]>([]);

	/** Currently active conversation */
	activeConversation = $state<DatabaseConversation | null>(null);

	/** Messages in the active conversation (filtered by currNode path) */
	activeMessages = $state<DatabaseMessage[]>([]);

	/** Whether the store has been initialized */
	isInitialized = $state(false);

	/** Pending MCP server overrides for new conversations (before first message) */
	pendingMcpServerOverrides = $state<McpServerOverride[]>([]);

	/** Callback for title update confirmation dialog */
	titleUpdateConfirmationCallback?: (currentTitle: string, newTitle: string) => Promise<boolean>;

	/**
	 * Modalities used in the active conversation.
	 * Computed from attachments in activeMessages.
	 */
	usedModalities: ModelModalities = $derived.by(() => {
		return this.calculateModalitiesFromMessages(this.activeMessages);
	});

	/** Reference to the client (lazy loaded to avoid circular dependency) */
	private _client: typeof import('$lib/clients/conversations.client').conversationsClient | null =
		null;

	private get client() {
		return this._client;
	}

	/** Check if store is ready (client initialized) */
	get isReady(): boolean {
		return this._client !== null;
	}

	/**
	 * Initialize the store by wiring up to the client.
	 * Must be called once after app startup.
	 */
	async init(): Promise<void> {
		if (!browser) return;
		if (this._client) return;

		const { conversationsClient } = await import('$lib/clients/conversations.client');
		this._client = conversationsClient;

		conversationsClient.setStoreCallbacks({
			getConversations: () => this.conversations,
			setConversations: (conversations) => (this.conversations = conversations),
			getActiveConversation: () => this.activeConversation,
			setActiveConversation: (conversation) => (this.activeConversation = conversation),
			getActiveMessages: () => this.activeMessages,
			setActiveMessages: (messages) => (this.activeMessages = messages),
			updateActiveMessages: (updater) => (this.activeMessages = updater(this.activeMessages)),
			setInitialized: (initialized) => (this.isInitialized = initialized),
			getPendingMcpServerOverrides: () => this.pendingMcpServerOverrides,
			setPendingMcpServerOverrides: (overrides) => (this.pendingMcpServerOverrides = overrides),
			getTitleUpdateConfirmationCallback: () => this.titleUpdateConfirmationCallback
		});

		await conversationsClient.initialize();
	}

	/**
	 * Calculate modalities from a list of messages.
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
	 */
	getModalitiesUpToMessage(messageId: string): ModelModalities {
		const messageIndex = this.activeMessages.findIndex((m) => m.id === messageId);

		if (messageIndex === -1) {
			return this.usedModalities;
		}

		const messagesBefore = this.activeMessages.slice(0, messageIndex);
		return this.calculateModalitiesFromMessages(messagesBefore);
	}

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

	async initialize(): Promise<void> {
		if (!this.client) return;
		return this.client.initialize();
	}

	async loadConversations(): Promise<void> {
		if (!this.client) return;
		return this.client.loadConversations();
	}

	async createConversation(name?: string): Promise<string> {
		if (!this.client) throw new Error('ConversationsStore not initialized');
		return this.client.createConversation(name);
	}

	async loadConversation(convId: string): Promise<boolean> {
		if (!this.client) return false;
		return this.client.loadConversation(convId);
	}

	clearActiveConversation(): void {
		if (!this.client) return;
		this.client.clearActiveConversation();
	}

	async deleteConversation(convId: string): Promise<void> {
		if (!this.client) return;
		return this.client.deleteConversation(convId);
	}

	async deleteAll(): Promise<void> {
		if (!this.client) return;
		return this.client.deleteAll();
	}

	async refreshActiveMessages(): Promise<void> {
		if (!this.client) return;
		return this.client.refreshActiveMessages();
	}

	async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		if (!this.client) return [];
		return this.client.getConversationMessages(convId);
	}

	async updateConversationName(convId: string, name: string): Promise<void> {
		if (!this.client) return;
		return this.client.updateConversationName(convId, name);
	}

	async updateConversationTitleWithConfirmation(
		convId: string,
		newTitle: string
	): Promise<boolean> {
		if (!this.client) return false;
		return this.client.updateConversationTitleWithConfirmation(convId, newTitle);
	}

	updateConversationTimestamp(): void {
		if (!this.client) return;
		this.client.updateConversationTimestamp();
	}

	async updateCurrentNode(nodeId: string): Promise<void> {
		if (!this.client) return;
		return this.client.updateCurrentNode(nodeId);
	}

	async navigateToSibling(siblingId: string): Promise<void> {
		if (!this.client) return;
		return this.client.navigateToSibling(siblingId);
	}

	getMcpServerOverride(serverId: string): McpServerOverride | undefined {
		if (!this.client) {
			return this.pendingMcpServerOverrides.find((o) => o.serverId === serverId);
		}
		return this.client.getMcpServerOverride(serverId);
	}

	isMcpServerEnabledForChat(serverId: string, globalEnabled: boolean): boolean {
		if (!this.client) {
			const override = this.pendingMcpServerOverrides.find((o) => o.serverId === serverId);
			return override !== undefined ? override.enabled : globalEnabled;
		}
		return this.client.isMcpServerEnabledForChat(serverId, globalEnabled);
	}

	async setMcpServerOverride(serverId: string, enabled: boolean | undefined): Promise<void> {
		if (!this.client) return;
		return this.client.setMcpServerOverride(serverId, enabled);
	}

	async toggleMcpServerForChat(serverId: string, globalEnabled: boolean): Promise<void> {
		if (!this.client) return;
		return this.client.toggleMcpServerForChat(serverId, globalEnabled);
	}

	async resetMcpServerToGlobal(serverId: string): Promise<void> {
		if (!this.client) return;
		return this.client.resetMcpServerToGlobal(serverId);
	}

	clearPendingMcpServerOverrides(): void {
		if (!this.client) {
			this.pendingMcpServerOverrides = [];
			return;
		}
		this.client.clearPendingMcpServerOverrides();
	}

	async downloadConversation(convId: string): Promise<void> {
		if (!this.client) return;
		return this.client.downloadConversation(convId);
	}

	async exportAllConversations(): Promise<DatabaseConversation[]> {
		if (!this.client) return [];
		return this.client.exportAllConversations();
	}

	async importConversations(): Promise<DatabaseConversation[]> {
		if (!this.client) return [];
		return this.client.importConversations();
	}

	async importConversationsData(
		data: ExportedConversations
	): Promise<{ imported: number; skipped: number }> {
		if (!this.client) return { imported: 0, skipped: 0 };
		return this.client.importConversationsData(data);
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
export const usedModalities = () => conversationsStore.usedModalities;
