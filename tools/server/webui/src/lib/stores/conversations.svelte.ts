/**
 * conversationsStore - Reactive State Store for Conversations
 *
 * Manages conversation lifecycle, persistence, navigation, and MCP server overrides.
 *
 * **Architecture & Relationships:**
 * - **DatabaseService**: Stateless IndexedDB layer
 * - **conversationsStore** (this): Reactive state + business logic
 * - **chatStore**: Chat-specific state (streaming, loading)
 *
 * **Key Responsibilities:**
 * - Conversation CRUD (create, load, delete)
 * - Message management and tree navigation
 * - MCP server per-chat overrides
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
import {
	filterByLeafNodeId,
	findLeafNode,
	runLegacyMigration,
	generateConversationTitle
} from '$lib/utils';
import type { McpServerOverride } from '$lib/types/database';
import { MessageRole, HtmlInputType, FileExtensionText } from '$lib/enums';
import {
	ISO_DATE_TIME_SEPARATOR,
	ISO_DATE_TIME_SEPARATOR_REPLACEMENT,
	ISO_TIMESTAMP_SLICE_LENGTH,
	EXPORT_CONV_ID_TRIM_LENGTH,
	EXPORT_CONV_NONALNUM_REPLACEMENT,
	EXPORT_CONV_NAME_SUFFIX_MAX_LENGTH,
	ISO_TIME_SEPARATOR,
	ISO_TIME_SEPARATOR_REPLACEMENT,
	NON_ALPHANUMERIC_REGEX,
	MULTIPLE_UNDERSCORE_REGEX,
	MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY
} from '$lib/constants';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

export interface ConversationTreeItem {
	conversation: DatabaseConversation;
	depth: number;
}

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

	/** Pending MCP server overrides for new conversations (before first message) */
	pendingMcpServerOverrides = $state<McpServerOverride[]>(ConversationsStore.loadMcpDefaults());

	/** Load MCP default overrides from localStorage */
	private static loadMcpDefaults(): McpServerOverride[] {
		if (typeof globalThis.localStorage === 'undefined') return [];
		try {
			const raw = localStorage.getItem(MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY);
			if (!raw) return [];
			const parsed = JSON.parse(raw);
			if (!Array.isArray(parsed)) return [];
			return parsed.filter(
				(o: unknown) => typeof o === 'object' && o !== null && 'serverId' in o && 'enabled' in o
			) as McpServerOverride[];
		} catch {
			return [];
		}
	}

	/** Persist MCP default overrides to localStorage */
	private saveMcpDefaults(): void {
		if (typeof globalThis.localStorage === 'undefined') return;
		const plain = this.pendingMcpServerOverrides.map((o) => ({
			serverId: o.serverId,
			enabled: o.enabled
		}));
		if (plain.length > 0) {
			localStorage.setItem(MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY, JSON.stringify(plain));
		} else {
			localStorage.removeItem(MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY);
		}
	}

	/** Callback for title update confirmation dialog */
	titleUpdateConfirmationCallback?: (currentTitle: string, newTitle: string) => Promise<boolean>;

	/**
	 * Callback for updating message content in chatStore.
	 * Registered by chatStore to enable cross-store updates without circular dependency.
	 */
	private messageUpdateCallback:
		| ((messageId: string, updates: Partial<DatabaseMessage>) => void)
		| null = null;

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
			// @deprecated Legacy migration for old marker-based messages.
			// Remove once all users have migrated to the structured format.
			await runLegacyMigration();

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
	 * Register a callback for message updates from other stores.
	 * Called by chatStore during initialization.
	 */
	registerMessageUpdateCallback(
		callback: (messageId: string, updates: Partial<DatabaseMessage>) => void
	): void {
		this.messageUpdateCallback = callback;
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

		if (this.pendingMcpServerOverrides.length > 0) {
			// Deep clone to plain objects (Svelte 5 $state uses Proxies which can't be cloned to IndexedDB)
			const plainOverrides = this.pendingMcpServerOverrides.map((o) => ({
				serverId: o.serverId,
				enabled: o.enabled
			}));
			conversation.mcpServerOverrides = plainOverrides;
			await DatabaseService.updateConversation(conversation.id, {
				mcpServerOverrides: plainOverrides
			});
			this.pendingMcpServerOverrides = [];
		}

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

			this.pendingMcpServerOverrides = [];
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
		// reload MCP defaults so new chats inherit persisted state
		this.pendingMcpServerOverrides = ConversationsStore.loadMcpDefaults();
	}

	/**
	 * Deletes a conversation and all its messages
	 * @param convId - The conversation ID to delete
	 */
	async deleteConversation(convId: string, options?: { deleteWithForks?: boolean }): Promise<void> {
		try {
			await DatabaseService.deleteConversation(convId, options);

			if (options?.deleteWithForks) {
				// Collect all descendants recursively
				const idsToRemove = new SvelteSet([convId]);
				const queue = [convId];
				while (queue.length > 0) {
					const parentId = queue.pop()!;
					for (const c of this.conversations) {
						if (c.forkedFromConversationId === parentId && !idsToRemove.has(c.id)) {
							idsToRemove.add(c.id);
							queue.push(c.id);
						}
					}
				}
				this.conversations = this.conversations.filter((c) => !idsToRemove.has(c.id));

				if (this.activeConversation && idsToRemove.has(this.activeConversation.id)) {
					this.clearActiveConversation();
					await goto(`?new_chat=true#/`);
				}
			} else {
				// Reparent direct children to deleted conv's parent (or promote to top-level)
				const deletedConv = this.conversations.find((c) => c.id === convId);
				const newParent = deletedConv?.forkedFromConversationId;
				this.conversations = this.conversations
					.filter((c) => c.id !== convId)
					.map((c) =>
						c.forkedFromConversationId === convId
							? { ...c, forkedFromConversationId: newParent }
							: c
					);

				if (this.activeConversation?.id === convId) {
					this.clearActiveConversation();
					await goto(`?new_chat=true#/`);
				}
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
					generateConversationTitle(
						newFirstUserMessage.content,
						Boolean(config().titleGenerationUseFirstLine)
					)
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
		if (this.activeConversation) {
			return this.activeConversation.mcpServerOverrides?.find(
				(o: McpServerOverride) => o.serverId === serverId
			);
		}
		return this.pendingMcpServerOverrides.find((o) => o.serverId === serverId);
	}

	/**
	 * Get all MCP server overrides for the current conversation.
	 * Returns pending overrides if no active conversation.
	 */
	getAllMcpServerOverrides(): McpServerOverride[] {
		if (this.activeConversation?.mcpServerOverrides) {
			return this.activeConversation.mcpServerOverrides;
		}
		return this.pendingMcpServerOverrides;
	}

	/**
	 * Checks if an MCP server is enabled for the active conversation.
	 * @param serverId - The server ID to check
	 * @returns True if server is enabled for this conversation
	 */
	isMcpServerEnabledForChat(serverId: string): boolean {
		const override = this.getMcpServerOverride(serverId);
		return override?.enabled ?? false;
	}

	/**
	 * Sets or removes MCP server override for the active conversation.
	 * If no conversation exists, stores as pending override.
	 * @param serverId - The server ID to override
	 * @param enabled - The enabled state, or undefined to remove override
	 */
	async setMcpServerOverride(serverId: string, enabled: boolean | undefined): Promise<void> {
		if (!this.activeConversation) {
			this.setPendingMcpServerOverride(serverId, enabled);
			return;
		}

		// Clone to plain objects to avoid Proxy serialization issues with IndexedDB
		const currentOverrides = (this.activeConversation.mcpServerOverrides || []).map(
			(o: McpServerOverride) => ({
				serverId: o.serverId,
				enabled: o.enabled
			})
		);
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

		await DatabaseService.updateConversation(this.activeConversation.id, {
			mcpServerOverrides: newOverrides.length > 0 ? newOverrides : undefined
		});

		this.activeConversation = {
			...this.activeConversation,
			mcpServerOverrides: newOverrides.length > 0 ? newOverrides : undefined
		};

		const convIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);
		if (convIndex !== -1) {
			this.conversations[convIndex].mcpServerOverrides =
				newOverrides.length > 0 ? newOverrides : undefined;
			this.conversations = [...this.conversations];
		}
	}

	/**
	 * Sets or removes a pending MCP server override (for new conversations).
	 */
	private setPendingMcpServerOverride(serverId: string, enabled: boolean | undefined): void {
		if (enabled === undefined) {
			this.pendingMcpServerOverrides = this.pendingMcpServerOverrides.filter(
				(o) => o.serverId !== serverId
			);
		} else {
			const existingIndex = this.pendingMcpServerOverrides.findIndex(
				(o) => o.serverId === serverId
			);
			if (existingIndex >= 0) {
				const newOverrides = [...this.pendingMcpServerOverrides];
				newOverrides[existingIndex] = { serverId, enabled };
				this.pendingMcpServerOverrides = newOverrides;
			} else {
				this.pendingMcpServerOverrides = [...this.pendingMcpServerOverrides, { serverId, enabled }];
			}
		}
		this.saveMcpDefaults();
	}

	/**
	 * Toggles MCP server enabled state for the active conversation.
	 * @param serverId - The server ID to toggle
	 */
	async toggleMcpServerForChat(serverId: string): Promise<void> {
		const currentEnabled = this.isMcpServerEnabledForChat(serverId);
		await this.setMcpServerOverride(serverId, !currentEnabled);
	}

	/**
	 * Removes MCP server override for the active conversation.
	 * @param serverId - The server ID to remove override for
	 */
	async removeMcpServerOverride(serverId: string): Promise<void> {
		await this.setMcpServerOverride(serverId, undefined);
	}

	/**
	 * Clears all pending MCP server overrides.
	 */
	clearPendingMcpServerOverrides(): void {
		this.pendingMcpServerOverrides = [];
		this.saveMcpDefaults();
	}

	/**
	 * Forks a conversation at a specific message, creating a new conversation
	 * containing messages from root up to the target message, then navigates to it.
	 *
	 * @param messageId - The message ID to fork at
	 * @param options - Fork options (name and whether to include attachments)
	 * @returns The new conversation ID, or null if fork failed
	 */
	async forkConversation(
		messageId: string,
		options: { name: string; includeAttachments: boolean }
	): Promise<string | null> {
		if (!this.activeConversation) return null;

		try {
			const newConv = await DatabaseService.forkConversation(
				this.activeConversation.id,
				messageId,
				options
			);

			this.conversations = [newConv, ...this.conversations];

			await goto(`#/chat/${newConv.id}`);

			toast.success('Conversation forked');

			return newConv.id;
		} catch (error) {
			console.error('Failed to fork conversation:', error);
			toast.error('Failed to fork conversation');

			return null;
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
	 * Generates a sanitized filename for a conversation export
	 * @param conversation - The conversation metadata
	 * @param msgs - Optional array of messages belonging to the conversation
	 * @returns The generated filename string
	 */
	generateConversationFilename(
		conversation: { id?: string; name?: string },
		msgs?: DatabaseMessage[]
	): string {
		const conversationName = (conversation.name ?? '').trim().toLowerCase();

		const sanitizedName = conversationName
			.replace(NON_ALPHANUMERIC_REGEX, EXPORT_CONV_NONALNUM_REPLACEMENT)
			.replace(MULTIPLE_UNDERSCORE_REGEX, '_')
			.substring(0, EXPORT_CONV_NAME_SUFFIX_MAX_LENGTH);

		// If we have messages, use the timestamp of the newest message
		const referenceDate = msgs?.length
			? new Date(Math.max(...msgs.map((m) => m.timestamp)))
			: new Date();

		const iso = referenceDate.toISOString().slice(0, ISO_TIMESTAMP_SLICE_LENGTH);
		const formattedDate = iso
			.replace(ISO_DATE_TIME_SEPARATOR, ISO_DATE_TIME_SEPARATOR_REPLACEMENT)
			.replaceAll(ISO_TIME_SEPARATOR, ISO_TIME_SEPARATOR_REPLACEMENT);
		const trimmedConvId = conversation.id?.slice(0, EXPORT_CONV_ID_TRIM_LENGTH) ?? '';
		return `${formattedDate}_conv_${trimmedConvId}_${sanitizedName}.json`;
	}

	/**
	 * Triggers a browser download of the provided exported conversation data
	 * @param data - The exported conversation payload (either a single conversation or array of them)
	 * @param filename - Filename; if omitted, a deterministic name is generated
	 */
	downloadConversationFile(data: ExportedConversations, filename?: string): void {
		// Choose the first conversation or message
		const conversation =
			'conv' in data ? data.conv : Array.isArray(data) ? data[0]?.conv : undefined;
		const msgs =
			'messages' in data ? data.messages : Array.isArray(data) ? data[0]?.messages : undefined;

		if (!conversation) {
			console.error('Invalid data: missing conversation');
			return;
		}

		let downloadFilename: string;

		if (filename) {
			downloadFilename = filename;
		} else if (Array.isArray(data) && data.length > 1) {
			downloadFilename = `${new Date().toISOString().split(ISO_DATE_TIME_SEPARATOR)[0]}_conversations.json`;
		} else {
			downloadFilename = this.generateConversationFilename(conversation, msgs);
		}

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

		this.downloadConversationFile({ conv: conversation, messages });
	}

	/**
	 * Imports conversations from a JSON file
	 * Opens file picker and processes the selected file
	 * @returns The list of imported conversations
	 */
	async importConversations(): Promise<DatabaseConversation[]> {
		return new Promise((resolve, reject) => {
			const input = document.createElement('input');
			input.type = HtmlInputType.FILE;
			input.accept = FileExtensionText.JSON;

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

/**
 * Builds a flat tree of conversations with depth levels for nested forks.
 * Accepts a pre-filtered list so search filtering stays in the component.
 */
export function buildConversationTree(convs: DatabaseConversation[]): ConversationTreeItem[] {
	const childrenByParent = new SvelteMap<string, DatabaseConversation[]>();
	const forkIds = new SvelteSet<string>();

	for (const conv of convs) {
		if (conv.forkedFromConversationId) {
			forkIds.add(conv.id);

			const siblings = childrenByParent.get(conv.forkedFromConversationId) || [];

			siblings.push(conv);
			childrenByParent.set(conv.forkedFromConversationId, siblings);
		}
	}

	const result: ConversationTreeItem[] = [];
	const visited = new SvelteSet<string>();

	function walk(conv: DatabaseConversation, depth: number) {
		visited.add(conv.id);
		result.push({ conversation: conv, depth });

		const children = childrenByParent.get(conv.id);
		if (children) {
			children.sort((a, b) => b.lastModified - a.lastModified);

			for (const child of children) {
				walk(child, depth + 1);
			}
		}
	}

	const roots = convs.filter((c) => !forkIds.has(c.id));
	for (const root of roots) {
		walk(root, 0);
	}

	for (const conv of convs) {
		if (!visited.has(conv.id)) {
			walk(conv, 1);
		}
	}

	return result;
}
