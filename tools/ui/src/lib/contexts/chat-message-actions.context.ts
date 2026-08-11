import { CONTEXT_KEY_CHAT_MESSAGE_ACTIONS } from '$lib/constants';
import { getContext, setContext } from 'svelte';

export interface ChatMessageDeletionInfo {
	totalCount: number;
	userMessages: number;
	assistantMessages: number;
	messageTypes: string[];
}

export interface ChatMessageActionsContext {
	readonly siblingInfo: ChatMessageSiblingInfo | null;
	readonly deletionInfo: ChatMessageDeletionInfo | null;
	readonly showDeleteDialog: boolean;
	copy: () => void;
	requestDelete: () => void;
	confirmDelete: () => void;
	setShowDeleteDialog: (show: boolean) => void;
	navigateToSibling: (siblingId: string) => void;
	forkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
}

const CHAT_MESSAGE_ACTIONS_KEY = Symbol.for(CONTEXT_KEY_CHAT_MESSAGE_ACTIONS);

/**
 * Sets the per-message actions context. Call this in the parent component (ChatMessage.svelte).
 */
export function setChatMessageActionsContext(
	ctx: ChatMessageActionsContext
): ChatMessageActionsContext {
	return setContext(CHAT_MESSAGE_ACTIONS_KEY, ctx);
}

/**
 * Gets the per-message actions context. Call this in child components.
 */
export function getChatMessageActionsContext(): ChatMessageActionsContext {
	return getContext(CHAT_MESSAGE_ACTIONS_KEY);
}
