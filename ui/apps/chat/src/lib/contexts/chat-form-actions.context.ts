import type { ChatFormActionsContext } from '#lib/types/index.ts';
import { createContext } from 'svelte';

/** The chat form actions: set in the parent component (ChatFormActions.svelte), read in its children. */
export const [getChatFormActionsContext, setChatFormActionsContext] =
	createContext<ChatFormActionsContext>();
