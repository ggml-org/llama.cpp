/**
 * Slash-command token detection for the chat form. Valid only at offset 0.
 */
export function findCommandToken(
	value: string
): { name: string; args: string; end: number } | null {
	if (!value.startsWith('/')) return null;

	const rest = value.slice(1);
	const spaceIdx = rest.search(/\s/);
	const name = spaceIdx === -1 ? rest : rest.slice(0, spaceIdx);
	const args = spaceIdx === -1 ? '' : rest.slice(spaceIdx + 1);

	return { args, end: value.length, name };
}

/**
 * Stable signature of a slash-command token for use as a "dismissed"
 * marker: while the picker is closed and this exact token is still intact,
 * the picker does not re-open on in-token edits.
 */
export interface CommandDismissSnapshot {
	name: string;
	args: string;
}

export function takeCommandDismissSnapshot(value: string): CommandDismissSnapshot | null {
	const token = findCommandToken(value);

	if (!token) return null;

	return { args: token.args, name: token.name };
}
