import { CONTEXT_KEY_CHAT_MESSAGE_ACTIONS } from '$lib/constants';
import type { ChatMessageActionsContext } from '$lib/types';
import { getContext, setContext } from 'svelte';

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
