import { CONTEXT_KEY_CHAT_MESSAGE_EDIT } from '$lib/constants';
import type { ChatMessageEditContext } from '$lib/types';
import { getContext, setContext } from 'svelte';

const CHAT_MESSAGE_EDIT_KEY = Symbol.for(CONTEXT_KEY_CHAT_MESSAGE_EDIT);

/**
 * Sets the message edit context. Call this in the parent component (ChatMessage.svelte).
 */
export function setChatMessageEditContext(ctx: ChatMessageEditContext): ChatMessageEditContext {
	return setContext(CHAT_MESSAGE_EDIT_KEY, ctx);
}

/**
 * Gets the message edit context. Call this in child components.
 */
export function getChatMessageEditContext(): ChatMessageEditContext {
	return getContext(CHAT_MESSAGE_EDIT_KEY);
}
