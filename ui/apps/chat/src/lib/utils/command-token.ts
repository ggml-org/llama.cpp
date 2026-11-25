import {
	CWD_COMMAND_LINE_BREAK,
	CWD_COMMAND_NAME,
	CWD_COMMAND_PREFIX
} from '#lib/constants/index.ts';

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

/**
 * The `/cwd` line at the top of the chat input: the path being typed after `/cwd `, and the
 * message below it, which the command leaves alone. Null when the input does not start with
 * the command.
 */
export interface CwdCommandLine {
	/** The path typed after `/cwd `, up to the end of the first line. */
	query: string;
	/** The text after the first line, without the newline between them. */
	message: string;
}

export function parseCwdCommandLine(value: string): CwdCommandLine | null {
	const token = findCommandToken(value);

	if (!token || token.name !== CWD_COMMAND_NAME) return null;

	const lineEnd = value.indexOf(CWD_COMMAND_LINE_BREAK);
	const line = lineEnd === -1 ? value : value.slice(0, lineEnd);
	const message = lineEnd === -1 ? '' : value.slice(lineEnd + 1);

	return { message, query: line.slice(CWD_COMMAND_PREFIX.length) };
}

/** The input with a `/cwd <query>` line above `message`. */
export function composeCwdCommandLine(query: string, message: string): string {
	const line = `${CWD_COMMAND_PREFIX}${query}`;

	return message ? `${line}${CWD_COMMAND_LINE_BREAK}${message}` : line;
}

/** Where the caret goes to edit the path: the end of the `/cwd` line. */
export function cwdCommandCaretOffset(query: string): number {
	return CWD_COMMAND_PREFIX.length + query.length;
}
