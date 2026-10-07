import { CONTEXT_KEY_CHAT_FORM_ACTIONS } from '$lib/constants';
import type { ChatFormActionsContext } from '$lib/types';
import { getContext, setContext } from 'svelte';

const CHAT_FORM_ACTIONS_KEY = Symbol.for(CONTEXT_KEY_CHAT_FORM_ACTIONS);

/**
 * Sets the chat form actions context. Call in the parent component (ChatFormActions.svelte).
 */
export function setChatFormActionsContext(ctx: ChatFormActionsContext): ChatFormActionsContext {
	return setContext(CHAT_FORM_ACTIONS_KEY, ctx);
}

/**
 * Gets the chat form actions context. Call in child components.
 */
export function getChatFormActionsContext(): ChatFormActionsContext {
	return getContext(CHAT_FORM_ACTIONS_KEY);
}
