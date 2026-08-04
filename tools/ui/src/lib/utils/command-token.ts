/**
 * Slash-command token detection for the chat form.
 *
 * Unlike the `@`-mention token, a slash command is only valid at the very
 * start of the input (offset 0) - `/foo` is a command, `hello /foo` is not.
 * The command name is the run of non-whitespace characters after `/`; the
 * args are everything after the first whitespace.
 *
 * Examples:
 *   `/`            -> { name: '', args: '', end: 1 }
 *   `/prompt`      -> { name: 'prompt', args: '', end: 7 }
 *   `/prompt rev`  -> { name: 'prompt', args: 'rev', end: 12 }
 *   `hello /prompt`-> null (not at offset 0)
 */
export function findCommandToken(
	value: string
): { name: string; args: string; end: number } | null {
	if (!value.startsWith('/')) return null;

	const rest = value.slice(1);
	const spaceIdx = rest.search(/\s/);
	const name = spaceIdx === -1 ? rest : rest.slice(0, spaceIdx);
	const args = spaceIdx === -1 ? '' : rest.slice(spaceIdx + 1);

	return { name, args, end: value.length };
}

/**
 * Stable signature of a slash-command token for use as a "dismissed"
 * marker. When the user hits Escape, the command picker records this so
 * subsequent in-token edits (typing more chars into `/prompt`) do not
 * silently re-open the picker or instant-dispatch. The signature changes
 * as soon as the user edits or deletes any character of the token, which
 * is the moment the picker is allowed to act again.
 */
export interface CommandDismissSnapshot {
	name: string;
	args: string;
}

export function takeCommandDismissSnapshot(value: string): CommandDismissSnapshot | null {
	const token = findCommandToken(value);
	if (!token) return null;
	return { name: token.name, args: token.args };
}
