import { getContext, setContext } from 'svelte';

export interface MessageEditState {
	readonly isEditing: boolean;
	readonly editedContent: string;
	readonly editedExtras: DatabaseMessageExtra[];
	readonly editedUploadedFiles: ChatUploadedFile[];
	readonly originalContent: string;
	readonly originalExtras: DatabaseMessageExtra[];
	readonly showSaveOnlyOption: boolean;
}

export interface MessageEditActions {
	setContent: (content: string) => void;
	setExtras: (extras: DatabaseMessageExtra[]) => void;
	setUploadedFiles: (files: ChatUploadedFile[]) => void;
	save: () => void;
	saveOnly: () => void;
	cancel: () => void;
	startEdit: () => void;
}

export type MessageEditContext = MessageEditState & MessageEditActions;

const MESSAGE_EDIT_KEY = Symbol.for('chat-message-edit');

/**
 * Sets the message edit context. Call this in the parent component (ChatMessage.svelte).
 */
export function setMessageEditContext(ctx: MessageEditContext): MessageEditContext {
	return setContext(MESSAGE_EDIT_KEY, ctx);
}

/**
 * Gets the message edit context. Call this in child components.
 */
export function getMessageEditContext(): MessageEditContext {
	return getContext(MESSAGE_EDIT_KEY);
}
