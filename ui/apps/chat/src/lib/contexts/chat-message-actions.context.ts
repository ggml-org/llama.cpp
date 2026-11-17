import type { ChatMessageActionsContext } from '#lib/types/index.ts';
import { createContext } from 'svelte';

/** The per-message actions: set in the parent component (ChatMessage.svelte), read in its children. */
export const [getChatMessageActionsContext, setChatMessageActionsContext] =
	createContext<ChatMessageActionsContext>();
