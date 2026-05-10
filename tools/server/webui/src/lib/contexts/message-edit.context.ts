import { getContext, setContext } from 'svelte';
import { CONTEXT_KEY_MESSAGE_EDIT } from '$lib/constants';
import { MessageRole } from '$lib/enums';

export interface MessageEditState {
	readonly isEditing: boolean;
	readonly editedContent: string;
	readonly editedExtras: DatabaseMessageExtra[];
	readonly editedUploadedFiles: ChatUploadedFile[];
	readonly originalContent: string;
	readonly originalExtras: DatabaseMessageExtra[];
	readonly showSaveOnlyOption: boolean;
	readonly showBranchAfterEditOption: boolean;
	readonly shouldBranchAfterEdit: boolean;
	readonly messageRole: MessageRole;
	readonly rawEditContent?: string;
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

export interface AssistantEditActions {
	setShouldBranchAfterEdit: (value: boolean) => void;
}

export type MessageEditContext = MessageEditState &
	MessageEditActions &
	Partial<AssistantEditActions>;

const MESSAGE_EDIT_KEY = Symbol.for(CONTEXT_KEY_MESSAGE_EDIT);

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
