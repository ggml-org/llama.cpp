/**
 * conversationsStore - Conversation lifecycle, persistence and navigation
 *
 * Owns conversation CRUD, message tree navigation, import/export and title
 * management, persisted through DatabaseService. Per-chat options (MCP
 * overrides, reasoning effort, cwd) live in ConversationPreferences,
 * composed as {@link ConversationsStore.preferences}.
 */

import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { ROUTES } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import { ConversationTransferService } from '$lib/services/conversation-transfer.service';
import { DatabaseService } from '$lib/services/database.service';
import { MigrationService } from '$lib/services/migration.service';
import { RouterService } from '$lib/services/router.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import {
	ConversationPreferences,
	type ConversationsPreferencesHost
} from '$lib/stores/conversations/preferences.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { tabsStore } from '$lib/stores/tabs.svelte';
import { filterByLeafNodeId, findLeafNode, generateConversationTitle } from '$lib/utils';
import { SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

class ConversationsStore implements ConversationsPreferencesHost {
	/** Currently active conversation */
	activeConversation = $state<DatabaseConversation | null>(null);

	/** Messages in the active conversation (filtered by currNode path) */
	activeMessages = $state<DatabaseMessage[]>([]);

	/** List of all conversations */
	conversations = $state<DatabaseConversation[]>([]);

	/** Whether the store has been initialized */
	isInitialized = $state(false);

	/** Per-chat options (MCP overrides, reasoning effort, cwd), composed here. */
	private _preferences = new ConversationPreferences(this);

	/**
	 * Listeners notified with the ids of conversations that were deleted.
	 * Lets dependent stores (e.g. agenticStore) drop per-conversation state
	 * without introducing a circular import back into this store.
	 */
	private conversationDeletionListeners = new Set<(convIds: string[]) => void>();

	/** In-flight init run; shared by concurrent callers, reset on failure to allow retry */
	private initPromise: Promise<void> | null = null;

	/**
	 * Memo of the last findMessageIndex() lookup. Streaming calls it once per
	 * chunk for the same message, so a validated cache hit keeps that O(1)
	 * instead of a linear scan of activeMessages on every token.
	 */
	private lastMessageIndex: { id: string; index: number } | null = null;

	get preferences() {
		return this._preferences;
	}

	/**
	 * Adds a message to the active messages array
	 */
	addMessageToActive(message: DatabaseMessage): void {
		this.activeMessages.push(message);
	}

	/**
	 * Applies a field update to a conversation row, mirroring it into both the
	 * conversations list and the active conversation when it is the target.
	 * Shared by the rename/pin/preferences flows so no caller can forget to
	 * mirror one side.
	 */
	applyConversationUpdate(id: string, updates: Partial<DatabaseConversation>): void {
		const convIndex = this.conversations.findIndex((c) => c.id === id);

		if (convIndex !== -1) {
			const target = this.conversations[convIndex] as unknown as Record<string, unknown>;

			for (const [key, value] of Object.entries(updates)) {
				if (target[key] !== value) target[key] = value;
			}
		}

		if (this.activeConversation?.id === id) {
			this.activeConversation = { ...this.activeConversation, ...updates };
		}
	}

	/**
	 * Derives a conversation title from its first message content and applies
	 * it, honoring the title-generation setting. Shared by every flow that
	 * edits or creates the first user message.
	 */
	async applyTitleFromContent(convId: string, content: string): Promise<void> {
		await this.updateConversationName(
			convId,
			generateConversationTitle(content, Boolean(settingsStore.config.titleGenerationUseFirstLine))
		);
	}

	/**
	 * Deletes multiple conversations in sequence.
	 * Mirrors deleteConversation() per-id; navigates to the new-chat screen only
	 * if the currently-open chat was among the deleted ones.
	 * @param convIds - Conversation IDs to delete
	 */
	async bulkDeleteConversations(convIds: string[]): Promise<void> {
		if (convIds.length === 0) return;

		try {
			const idsToRemove = new SvelteSet(convIds);
			// Collect all descendants recursively so the local cache stays consistent
			// even when deleteWithForks is omitted.
			const queue = [...convIds];

			while (queue.length > 0) {
				const parentId = queue.pop()!;

				for (const c of this.conversations) {
					if (c.forkedFromConversationId === parentId && !idsToRemove.has(c.id)) {
						idsToRemove.add(c.id);
						queue.push(c.id);
					}
				}
			}

			const activeWasDeleted =
				this.activeConversation !== null && idsToRemove.has(this.activeConversation.id);

			await DatabaseService.bulkDeleteConversations([...idsToRemove]);

			this.conversations = this.conversations.filter((c) => !idsToRemove.has(c.id));
			this.notifyConversationsDeleted([...idsToRemove]);

			if (activeWasDeleted) {
				const activeId = this.activeConversation!.id;

				tabsStore.removeTabs([...idsToRemove].filter((id) => id !== activeId));
				this.clearActiveConversation();
				await tabsStore.close(activeId, activeId);
			} else {
				tabsStore.removeTabs([...idsToRemove]);
			}

			toast.success(
				idsToRemove.size === 1
					? 'Conversation deleted'
					: `${idsToRemove.size} conversations deleted`
			);
		} catch (error) {
			console.error('Failed to bulk delete conversations:', error);
			toast.error('Failed to delete conversations');
		}
	}

	/**
	 * Bundles the given conversations into a single zip archive and triggers a
	 * browser download (one JSONL file per conversation).
	 * @param convIds - Conversation IDs to export
	 */
	async bulkExportConversations(convIds: string[]): Promise<void> {
		if (convIds.length === 0) return;

		try {
			const fetched = await DatabaseService.getConversationsWithMessages(convIds);
			const activeId = this.activeConversation?.id;
			const overridden = fetched.get(activeId ?? '');

			if (overridden && activeId) {
				overridden.conv = { ...this.activeConversation! };
			}

			const exported = [...fetched.values()];

			if (exported.length === 0) {
				toast.error('No conversations to export');

				return;
			}

			ConversationTransferService.downloadConversationsArchive(exported);

			toast.success(
				exported.length === 1
					? 'Conversation exported'
					: `${exported.length} conversations exported`
			);
		} catch (error) {
			console.error('Failed to bulk export conversations:', error);
			toast.error('Failed to export conversations');
		}
	}

	/**
	 * Toggles the pinned state of each conversation individually.
	 * Mixed-pin selections are intentionally not normalised here; the bulk
	 * action UI surfaces them as a disabled mixed-state instead.
	 * @param convIds - Conversation IDs to toggle
	 */
	async bulkToggleConversationPin(convIds: string[]): Promise<void> {
		if (convIds.length === 0) return;

		try {
			const updates = await DatabaseService.bulkToggleConversationPins(convIds);
			const activeId = this.activeConversation?.id;

			if (activeId && updates.has(activeId)) {
				this.activeConversation = {
					...this.activeConversation!,
					pinned: updates.get(activeId)!
				};
			}

			for (let i = 0; i < this.conversations.length; i++) {
				const newPinned = updates.get(this.conversations[i].id);

				if (newPinned !== undefined) this.conversations[i].pinned = newPinned;
			}

			toast.success(
				convIds.length === 1
					? 'Conversation pin toggled'
					: `Updated pin state for ${convIds.length} conversations`
			);
		} catch (error) {
			console.error('Failed to bulk toggle pin:', error);
			toast.error('Failed to update pin state');
		}
	}

	/**
	 * Clears the active conversation and messages.
	 */
	clearActiveConversation(): void {
		this.activeConversation = null;
		this.activeMessages = [];
		// reload defaults so new chats inherit persisted state
		this.preferences.resetPending();
	}

	/**
	 * Creates a new conversation and navigates to it
	 * @param name - Optional name for the conversation
	 * @returns The ID of the created conversation
	 */
	async createConversation(name?: string): Promise<string> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;
		// Working directory and reasoning effort picked on the new-chat screen
		// get threaded into the new conversation here, then cleared so they
		// don't bleed onto subsequent new chats.
		const conversation = await DatabaseService.createConversation(conversationName, {
			cwd: this.preferences.pendingCwd ?? undefined,
			reasoningEffort: this.preferences.pendingReasoningEffort
		});

		this.preferences.pendingCwd = null;

		this.conversations = [conversation, ...this.conversations];
		this.activeConversation = conversation;
		this.activeMessages = [];

		await goto(RouterService.chat(conversation.id));

		return conversation.id;
	}

	/**
	 * Deletes all conversations and their messages
	 */
	async deleteAll(): Promise<void> {
		try {
			const allConversations = await DatabaseService.getAllConversations();
			const allIds = allConversations.map((c) => c.id);

			await DatabaseService.bulkDeleteConversations(allIds);

			this.clearActiveConversation();
			this.conversations = [];
			tabsStore.clear();
			this.notifyConversationsDeleted(allIds);

			toast.success('All conversations deleted');

			await goto(ROUTES.START);
		} catch (error) {
			console.error('Failed to delete all conversations:', error);
			toast.error('Failed to delete conversations');
		}
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
					const activeId = this.activeConversation.id;

					tabsStore.removeTabs([...idsToRemove].filter((id) => id !== activeId));
					this.clearActiveConversation();
					await tabsStore.close(activeId, activeId);
				} else {
					tabsStore.removeTabs([...idsToRemove]);
				}

				this.notifyConversationsDeleted([...idsToRemove]);
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
					await tabsStore.close(convId, convId);
				} else {
					tabsStore.removeTabs([convId]);
				}

				this.notifyConversationsDeleted([convId]);
			}
		} catch (error) {
			console.error('Failed to delete conversation:', error);
		}
	}

	/**
	 * Downloads a single conversation as a JSONL file, serializing the full message tree.
	 * @param convId - The conversation ID to download
	 */
	async downloadConversation(convId: string): Promise<void> {
		const conversation =
			this.activeConversation?.id === convId
				? this.activeConversation
				: await DatabaseService.getConversation(convId);

		if (!conversation) return;

		const messages = await DatabaseService.getConversationMessages(convId);

		ConversationTransferService.downloadConversationFile({ conv: conversation, messages });
	}

	/**
	 * Finds the index of a message in active messages.
	 *
	 * The last lookup is memoized and reused when it still validates against
	 * the current array (same id at the same position), which covers the
	 * streaming hot path where the same message is looked up on every chunk
	 * while the array itself only mutates by field. Any structural change
	 * (splice, reassignment, reordering) fails validation and falls back to a
	 * full scan.
	 */
	findMessageIndex(messageId: string): number {
		const last = this.lastMessageIndex;
		const messages = this.activeMessages;

		if (
			last &&
			last.id === messageId &&
			last.index >= 0 &&
			last.index < messages.length &&
			messages[last.index]?.id === messageId
		) {
			return last.index;
		}

		const index = messages.findIndex((m) => m.id === messageId);

		this.lastMessageIndex = { id: messageId, index };

		return index;
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

			await goto(RouterService.chat(newConv.id));

			toast.success('Conversation forked');

			return newConv.id;
		} catch (error) {
			console.error('Failed to fork conversation:', error);
			toast.error('Failed to fork conversation');

			return null;
		}
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
	 * @returns The conversations written to the database and the ones skipped
	 */
	async importConversationsData(
		data: ExportedConversations
	): Promise<{ imported: DatabaseConversation[]; skipped: DatabaseConversation[] }> {
		const result = await DatabaseService.importConversations(data);

		await this.loadConversations();

		return result;
	}

	/**
	 * Initialize the store by loading conversations from database.
	 * Safe to call multiple times: concurrent callers share a single run,
	 * and a failed run can be retried by calling again.
	 */
	initialize(): Promise<void> {
		if (!browser) return Promise.resolve();

		if (this.initPromise) return this.initPromise;

		this.initPromise = (async () => {
			try {
				await MigrationService.runAllMigrations();
				await this.loadConversations();
				this.isInitialized = true;
			} catch (error) {
				console.error('Failed to initialize conversations:', error);
				this.initPromise = null;
			}
		})();

		return this.initPromise;
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

			// Drop any cwd the user drafted on the empty new-chat screen -
			// it doesn't belong to this conversation.
			this.preferences.pendingCwd = null;

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
	 * Loads all conversations from the database
	 */
	async loadConversations(): Promise<void> {
		const conversations = await DatabaseService.getAllConversations();

		this.conversations = conversations;
	}

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
				await this.applyTitleFromContent(this.activeConversation.id, newFirstUserMessage.content);
			}
		}
	}

	/**
	 * Registers a listener invoked with the ids of deleted conversations.
	 * Returns an unsubscribe function.
	 */
	onConversationsDeleted(listener: (convIds: string[]) => void): () => void {
		this.conversationDeletionListeners.add(listener);

		return () => this.conversationDeletionListeners.delete(listener);
	}

	/**
	 * Start a fresh chat by navigating to the bare `#/` new-chat screen. The
	 * chat layout opens a new-chat tab for it when Conversation tabs are on.
	 */
	async openNewChat(): Promise<void> {
		this.clearActiveConversation();
		await goto(ROUTES.START);
	}

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
	 * Removes a message from active messages by index
	 */
	removeMessageAtIndex(index: number): DatabaseMessage | undefined {
		if (index !== -1) {
			return this.activeMessages.splice(index, 1)[0];
		}

		return undefined;
	}

	/**
	 * Removes messages from active messages starting at an index
	 */
	sliceActiveMessages(startIndex: number): void {
		this.activeMessages = this.activeMessages.slice(0, startIndex);
	}

	/**
	 * Toggles the pinned status of a conversation.
	 * @param convId - The conversation ID to toggle
	 * @returns The new pinned status
	 */
	async toggleConversationPin(convId: string): Promise<boolean> {
		try {
			const newPinnedState = await DatabaseService.toggleConversationPin(convId);

			this.applyConversationUpdate(convId, { pinned: newPinnedState });

			return newPinnedState;
		} catch (error) {
			console.error('Failed to toggle conversation pin:', error);

			return false;
		}
	}

	/**
	 * Updates the name of a conversation.
	 * @param convId - The conversation ID to update
	 * @param name - The new name for the conversation
	 */
	async updateConversationName(convId: string, name: string): Promise<void> {
		try {
			await DatabaseService.updateConversation(convId, { name });

			this.applyConversationUpdate(convId, { name });
		} catch (error) {
			console.error('Failed to update conversation name:', error);
		}
	}

	/**
	 * Marks a conversation as recently active: stamps lastModified (persisted)
	 * and moves it to the top of the list. Only message-activity flows call
	 * this; metadata updates (rename, pin, settings) do not.
	 *
	 * @param convId - Conversation that produced the activity, defaults to the active one
	 */
	updateConversationTimestamp(convId?: string): void {
		const targetId = convId ?? this.activeConversation?.id;

		if (!targetId) return;

		const now = Date.now();
		const chatIndex = this.conversations.findIndex((c) => c.id === targetId);

		if (chatIndex !== -1) {
			this.conversations[chatIndex].lastModified = now;
			const updatedConv = this.conversations.splice(chatIndex, 1)[0];

			this.conversations = [updatedConv, ...this.conversations];
		}

		if (this.activeConversation?.id === targetId) {
			this.activeConversation = { ...this.activeConversation, lastModified: now };
		}

		DatabaseService.updateConversation(targetId, { lastModified: now }).catch((error) =>
			console.error('Failed to update conversation timestamp:', error)
		);
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
	 * Updates a message at a specific index in active messages
	 */
	updateMessageAtIndex(index: number, updates: Partial<DatabaseMessage>): void {
		const message = index === -1 ? undefined : this.activeMessages[index];

		if (!message) return;

		// Assign field by field rather than replacing the object. Replacing it
		// changes the array slot, which invalidates every consumer that merely
		// walks the list - notably ChatMessages.displayMessages, which rebuilds
		// entries for every message in the conversation. Deep $state proxies make
		// per-field writes fine-grained, so only readers of the changed field wake.
		const target = message as unknown as Record<string, unknown>;

		for (const [key, value] of Object.entries(updates)) {
			if (target[key] !== value) {
				target[key] = value;
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

	private notifyConversationsDeleted(convIds: string[]): void {
		if (convIds.length === 0) return;

		for (const listener of this.conversationDeletionListeners) {
			listener(convIds);
		}
	}
}

export const conversationsStore = new ConversationsStore();
