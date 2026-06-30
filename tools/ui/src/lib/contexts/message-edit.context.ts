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

	/**
	 * System-message affordances surfaced to the edit form.
	 * All default to false when messageRole !== SYSTEM.
	 */
	readonly isSystemMessage: boolean;
	readonly isSystemPlaceholder: boolean;
	/** Available when the user may attach the system message to a library skill. */
	readonly canAddToLibrary: boolean;
	/** Available when the system message is already linked to an existing (non-MCP) library skill. */
	readonly canUpdateLibrarySkill: boolean;
	/** Title of the linked library skill, surfaced to the form for labeling. */
	readonly librarySkillTitle?: string;
}

export interface MessageEditActions {
	setContent: (content: string) => void;
	setExtras: (extras: DatabaseMessageExtra[]) => void;
	setUploadedFiles: (files: ChatUploadedFile[]) => void;
	/** Save the edit and, for system messages, route through the "Add to Skills library" dialog. */
	save: () => void;
	saveOnly: () => void;
	/** For system messages: same effect as `save` plus opens the add-to-library dialog. */
	saveAsLibrary: () => void;
	/** For system messages linked to a library skill: save and propagate new content to the skill. */
	updateLibraryPrompt: () => void;
	updateLibrarySkill?: () => void;
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
