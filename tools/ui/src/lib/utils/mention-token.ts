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

/**
 * Find the most-recent `@`-mention token whose extent includes `cursor`;
 * the query covers the whole `@...` token regardless of caret position.
 */
export function findMentionToken(
	value: string,
	cursor: number
): { start: number; end: number; query: string } | null {
	if (cursor <= 0 || cursor > value.length) return null;

	let atIndex = -1;
	for (let i = cursor - 1; i >= 0; i--) {
		const ch = value[i];
		if (ch === '@') {
			const prev = i > 0 ? value[i - 1] : '';
			if (i === 0 || TOKEN_BOUNDARY_CHARS.has(prev)) {
				atIndex = i;
			}
			break;
		}
		if (TOKEN_BOUNDARY_CHARS.has(ch)) break;
	}

	if (atIndex === -1) return null;

	let end = atIndex + 1;
	while (end < value.length && !TOKEN_BOUNDARY_CHARS.has(value[end])) {
		end++;
	}

	return {
		start: atIndex,
		end,
		query: value.slice(atIndex + 1, end)
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
	return { start: token.start, query: token.query };
}
