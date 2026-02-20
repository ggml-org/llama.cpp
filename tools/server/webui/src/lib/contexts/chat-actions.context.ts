import { getContext, setContext } from 'svelte';

export interface ChatActionsContext {
	copy: (message: DatabaseMessage) => void;
	delete: (message: DatabaseMessage) => void;
	navigateToSibling: (siblingId: string) => void;
	editWithBranching: (
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) => void;
	editWithReplacement: (
		message: DatabaseMessage,
		newContent: string,
		shouldBranch: boolean
	) => void;
	editUserMessagePreserveResponses: (
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) => void;
	regenerateWithBranching: (message: DatabaseMessage, modelOverride?: string) => void;
	continueAssistantMessage: (message: DatabaseMessage) => void;
}

const CHAT_ACTIONS_KEY = Symbol.for('chat-actions');

export function setChatActionsContext(ctx: ChatActionsContext): ChatActionsContext {
	return setContext(CHAT_ACTIONS_KEY, ctx);
}

export function getChatActionsContext(): ChatActionsContext {
	return getContext(CHAT_ACTIONS_KEY);
}
