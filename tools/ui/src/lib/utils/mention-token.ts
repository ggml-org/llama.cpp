/**
 * Token-boundary characters for the `@`-mention language. The mention starts
 * at a `@` whose previous character is either start-of-string or one of these
 * - typing `[` opens a markdown link, `,`/`;` separates list items,
 * whitespace splits words, and `(` follows function calls. The set is
 * conservative on purpose: `user@example.com` should NOT trigger because
 * the `@` is in the middle of an identifier.
 */
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
 * Find the most-recent `@`-mention token whose extent includes `cursor`.
 *
 * The mention token starts at the last `@` at-or-before `cursor` that sits
 * at a token boundary. The token ends at `cursor` - this matches the model
 * "while the user is typing inside the token, the picker is open".
 *
 * Returns `null` when the cursor is not currently inside a valid mention
 * (no `@` in range, the `@` is mid-identifier, or the cursor is before the
 * only `@`).
 *
 * Examples (cursor marked with `^`):
 *   `^`             -> null
 *   `@pr^`          -> { start: 0, end: 3, query: 'pr' }
 *   `hello @pr^`    -> { start: 6, end: 9, query: 'pr' }
 *   `em@^`          -> null (mid-identifier)
 *   `text@pr^`      -> null (mid-identifier; no space before `@`)
 *   `@pr hello^`    -> null (cursor is past the whitespace break)
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

	return {
		start: atIndex,
		end: cursor,
		query: value.slice(atIndex + 1, cursor)
	};
}

/**
 * Stable signature of a mention token for use as a "dismissed" marker.
 *
 * When the user hits Escape, the picker records this signature so that
 * subsequent in-token edits (typing more chars into `@hello`) do not
 * silently re-open the picker. The signature changes as soon as the user
 * edits or deletes any character of the token, which is the moment the
 * picker is allowed to re-open.
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
