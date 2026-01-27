/**
 * Shared Active Conversation State
 *
 * This module provides a dependency-free shared state for tracking the active conversation.
 * It eliminates circular dependencies between chatStore and conversationsStore.
 *
 * **Why this exists:**
 * - chatStore needs to know the active conversation ID to sync global loading/streaming state
 * - conversationsStore manages the active conversation
 * - Direct imports between stores would create circular dependencies
 *
 * **Usage:**
 * - conversationsStore: calls setId() when switching conversations
 * - chatStore: calls isActive() to check if state should sync to global
 */

class ActiveConversationStore {
	private _id = $state<string | null>(null);

	/**
	 * Get the currently active conversation ID.
	 * Returns null if no conversation is active.
	 */
	get id(): string | null {
		return this._id;
	}

	/**
	 * Set the active conversation ID.
	 * Should only be called by conversationsStore when switching conversations.
	 */
	setId(id: string | null): void {
		this._id = id;
	}

	/**
	 * Check if the given conversation ID is the currently active one.
	 */
	isActive(convId: string): boolean {
		return this._id === convId;
	}
}

export const activeConversationStore = new ActiveConversationStore();

// Convenience exports for backward compatibility
export const getActiveConversationId = () => activeConversationStore.id;
export const setActiveConversationId = (id: string | null) => activeConversationStore.setId(id);
export const isActiveConversation = (convId: string) => activeConversationStore.isActive(convId);
