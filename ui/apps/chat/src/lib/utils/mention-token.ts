// An `@` starts a mention only when preceded by start-of-string or one of
// these; identifier chars are not delimiters, so a mid-word `@` does not.
const TOKEN_BOUNDARY_CHARS = new Set([
	' ',
	'\t',
	'\n',
	'\r',
	'(',
	')',
	'[',
	']',
	',',
	';',
	':',
	'"',
	"'"
]);
const MENTION_TRIGGER = '@';
const SPACE = ' ';

/**
 * Find the most-recent `@`-mention token whose extent includes `cursor`.
 *
 * Names hold spaces (`@My Documents/notes`), so a single space before the caret is part of
 * the token: only two in a row, or another boundary character, end it. After the caret the
 * token runs to the end of the word, so picking a result never takes the text that follows.
 */
export function findMentionToken(
	value: string,
	cursor: number
): { start: number; end: number; query: string } | null {
	if (cursor <= 0 || cursor > value.length) return null;

	let atIndex = -1;

	for (let i = cursor - 1; i >= 0; i--) {
		const ch = value[i];
		const prev = i > 0 ? value[i - 1] : '';

		if (ch === MENTION_TRIGGER) {
			if (i === 0 || TOKEN_BOUNDARY_CHARS.has(prev)) {
				atIndex = i;
			}

			break;
		}

		if (ch === SPACE) {
			// two spaces end the token, and a space right after `@` never starts one
			if (prev === SPACE || prev === MENTION_TRIGGER) break;

			continue;
		}

		if (TOKEN_BOUNDARY_CHARS.has(ch)) break;
	}

	if (atIndex === -1) return null;

	let end = cursor;

	while (end < value.length && !TOKEN_BOUNDARY_CHARS.has(value[end])) {
		end++;
	}

	return {
		end,
		query: value.slice(atIndex + 1, end),
		start: atIndex
	};
}

/**
 * Stable signature of a mention token for use as a "dismissed" marker:
 * while the picker is closed and this exact token is still intact, the
 * picker does not silently re-open on in-token edits.
 */
export interface MentionDismissSnapshot {
	start: number;
	query: string;
}

export function takeMentionDismissSnapshot(
	value: string,
	cursor: number
): MentionDismissSnapshot | null {
	const token = findMentionToken(value, cursor);

	if (!token) return null;

	return { query: token.query, start: token.start };
}

/**
 * True while a dismissed mention is still the one at the caret: the same token, or that
 * token with more words typed after it, which is the message going on.
 */
export function isMentionDismissed(
	snapshot: MentionDismissSnapshot | null,
	token: { start: number; query: string }
): boolean {
	if (!snapshot || snapshot.start !== token.start) return false;

	return token.query === snapshot.query || token.query.startsWith(snapshot.query.trimEnd() + SPACE);
}
