import { ChatFormInputRichTokenKind } from '$lib/enums';

/**
 * A single token produced by the chat-form-input-rich tokenizer:
 * plain text, a file/folder mention badge, or an inline/fenced code span.
 */
export type ChatFormInputRichToken =
	| { kind: ChatFormInputRichTokenKind.TEXT; text: string }
	| { kind: ChatFormInputRichTokenKind.BADGE; name: string; path: string }
	| { kind: ChatFormInputRichTokenKind.CODE_INLINE; text: string }
	| { kind: ChatFormInputRichTokenKind.CODE_BLOCK; text: string };
