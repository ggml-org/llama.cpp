import type { ChatMessageEditContext } from '#lib/types/index.ts';
import { createContext } from 'svelte';

/** The message edit state: set in the parent component (ChatMessage.svelte), read in its children. */
export const [getChatMessageEditContext, setChatMessageEditContext] =
	createContext<ChatMessageEditContext>();
