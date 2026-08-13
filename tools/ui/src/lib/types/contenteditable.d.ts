import { ContentEditableTokenKind } from '$lib/enums';

/**
 * A single token produced by the chat-form contenteditable tokenizer:
 * plain text, a file/folder mention badge, or an inline/fenced code span.
 */
export type ContentEditableToken =
	| { kind: ContentEditableTokenKind.TEXT; text: string }
	| { kind: ContentEditableTokenKind.BADGE; name: string; path: string }
	| { kind: ContentEditableTokenKind.INLINE_CODE; text: string }
	| { kind: ContentEditableTokenKind.CODE_BLOCK; text: string };
