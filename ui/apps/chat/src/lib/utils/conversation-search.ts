/**
 * Matching for the search dialog's message search: which text of a message is searched and the
 * snippet shown for a match. No IndexedDB access here, the service feeds it messages.
 */

import {
	CONVERSATION_SEARCH_SNIPPET_LEAD,
	CONVERSATION_SEARCH_SNIPPET_MAX_CHARS
} from '#lib/constants/index.ts';
import type { DatabaseMessage } from '#lib/types/database.js';

export interface ConversationSearchSnippet {
	/** Text before the match, with a leading `...` when cut */
	before: string;
	match: string;
	/** Text after the match, with a trailing `...` when cut */
	after: string;
}

/** Tool call names and arguments, without the call ids */
function toolCallTexts(toolCalls: string): string[] {
	try {
		const parsed: unknown = JSON.parse(toolCalls);

		if (!Array.isArray(parsed)) return [];

		return parsed.flatMap((call) => {
			const fn = call?.function;

			if (!fn) return [];

			return [fn.name, fn.arguments].filter(
				(value): value is string => typeof value === 'string' && value !== ''
			);
		});
	} catch {
		return [];
	}
}

/**
 * Every text a person could remember from a message, in reading order: reasoning, the content,
 * tool calls and the names and text of attachments. Ids, timestamps and base64 payloads are left out.
 */
export function searchableTexts(message: DatabaseMessage): string[] {
	const texts: string[] = [];

	if (message.isSynthetic) return texts;

	if (message.reasoningContent) texts.push(message.reasoningContent);
	else if (message.thinking) texts.push(message.thinking);

	if (typeof message.content === 'string' && message.content) texts.push(message.content);

	if (message.toolCalls) texts.push(...toolCallTexts(message.toolCalls));

	for (const extra of message.extra ?? []) {
		if (extra.name) texts.push(extra.name);

		if ('content' in extra && typeof extra.content === 'string' && extra.content) {
			texts.push(extra.content);
		}
	}

	return texts;
}

/** First match of `needle` (already lowercased) in the message's text, or null */
export function findMessageMatch(
	message: DatabaseMessage,
	needle: string
): { at: number; text: string } | null {
	if (!needle) return null;

	for (const text of searchableTexts(message)) {
		const lower = text.toLowerCase();
		const at = lower.indexOf(needle);

		// lowercasing can change the length (e.g. U+0130), then the offset does not map back
		if (at !== -1) return { at: lower.length === text.length ? at : -1, text };
	}

	return null;
}

/** A one-line excerpt around the match, cut to `maxChars` with the match in it */
export function matchSnippet(
	text: string,
	at: number,
	length: number,
	maxChars: number = CONVERSATION_SEARCH_SNIPPET_MAX_CHARS
): ConversationSearchSnippet {
	const flat = (value: string) => value.replace(/\s+/g, ' ');

	if (at < 0) {
		const whole = flat(text).trim();

		return {
			after: whole.length > maxChars ? `${whole.slice(0, maxChars)}...` : whole,
			before: '',
			match: ''
		};
	}

	const lead = Math.min(at, CONVERSATION_SEARCH_SNIPPET_LEAD);
	const start = at - lead;
	const end = Math.min(text.length, start + Math.max(maxChars, lead + length));
	const before = flat(text.slice(start, at)).trimStart();
	const after = flat(text.slice(at + length, end)).trimEnd();

	return {
		after: end < text.length ? `${after}...` : after,
		before: start > 0 ? `...${before}` : before,
		match: flat(text.slice(at, at + length))
	};
}
