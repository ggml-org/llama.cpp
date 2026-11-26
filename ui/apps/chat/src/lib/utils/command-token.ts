import {
	COMMAND_NAME_REGEX,
	CWD_COMMAND_PREFIX,
	PROMPT_TRIGGER_PREFIX
} from '#lib/constants/index.ts';

const SPACE = ' ';
const WHITESPACE_REGEX = /\s/;
const COMMAND_NAME_PREFIX_REGEX = /^([\w-]+)/;

/** A slash command typed in the chat input: `/name`, and the argument after it. */
export interface CommandToken {
	/** Offset of the `/`. */
	start: number;
	/** Offset just past the token. */
	end: number;
	name: string;
	/** What follows the name and its space, e.g. the path of `/cwd ~/git`. */
	args: string;
}

const isWhitespace = (char: string | undefined) => !!char && WHITESPACE_REGEX.test(char);

/** True when a command can start at `index`: the input's start, or right after whitespace. */
function startsCommand(value: string, index: number): boolean {
	return value[index] === PROMPT_TRIGGER_PREFIX && (index === 0 || isWhitespace(value[index - 1]));
}

/** Where the text the caret is in begins: after a new line or two spaces in a row. */
function runStart(value: string, cursor: number): number {
	let i = cursor;

	while (i > 0) {
		const char = value[i - 1];

		if (isWhitespace(char) && char !== SPACE) break;

		if (char === SPACE && value[i - 2] === SPACE) break;

		i--;
	}

	return i;
}

/** Past the word the caret is in: the token never takes the text after it. */
function wordEnd(value: string, cursor: number): number {
	let end = cursor;

	while (end < value.length && !isWhitespace(value[end])) end++;

	return end;
}

/**
 * The slash command the caret is in, anywhere in the input: a `/` at the input's start or
 * after whitespace, then the command's name, as in `fix this /cwd ~/git`.
 *
 * A command that takes an argument (one of `commands`, or the start of one's name) takes
 * it after a space, single spaces included (`/cwd ~/My Documents`); like an `@` mention,
 * two spaces or a new line end it. So a path in the argument (`/cwd /usr/lo`) is not taken
 * for a command of its own, unless it names one. Any other command ends at its name: the
 * words after `/model` are the message's. After the caret the token runs to the end of the
 * word.
 */
export function findCommandToken(
	value: string,
	cursor: number,
	commands: readonly string[] = []
): CommandToken | null {
	if (cursor < 0 || cursor > value.length) return null;

	const end = wordEnd(value, cursor);
	const from = runStart(value, cursor);

	// the word the caret is in, when it is a `/name`
	let word: CommandToken | null = null;

	for (let i = cursor - 1; i >= from && !isWhitespace(value[i]); i--) {
		if (!startsCommand(value, i)) continue;

		const name = value.slice(i + 1, end);

		if (COMMAND_NAME_REGEX.test(name)) word = { args: '', end, name, start: i };

		break;
	}

	// a command that takes an argument further back on the run, with the caret in it
	let withArgs: CommandToken | null = null;

	for (let i = cursor - 1; i >= from; i--) {
		if (!startsCommand(value, i)) continue;

		// a known command, or the start of one (`/cw docs` completes to `/cwd docs`)
		const name = COMMAND_NAME_PREFIX_REGEX.exec(value.slice(i + PROMPT_TRIGGER_PREFIX.length))?.[1];

		if (!name || value[i + PROMPT_TRIGGER_PREFIX.length + name.length] !== SPACE) continue;

		if (!commands.some((command) => command.startsWith(name))) continue;

		const argsStart = i + PROMPT_TRIGGER_PREFIX.length + name.length + SPACE.length;

		if (argsStart > cursor) continue;

		withArgs = { args: value.slice(argsStart, end), end, name, start: i };

		break;
	}

	if (word && (!withArgs || commands.some((command) => command.startsWith(word.name)))) {
		return word;
	}

	return withArgs ?? word;
}

/**
 * Stable signature of a slash-command token for use as a "dismissed"
 * marker: while the picker is closed and this exact token is still intact,
 * the picker does not re-open on in-token edits.
 */
export interface CommandDismissSnapshot {
	start: number;
	name: string;
	args: string;
}

export function takeCommandDismissSnapshot(
	value: string,
	cursor: number,
	commands: readonly string[] = []
): CommandDismissSnapshot | null {
	const token = findCommandToken(value, cursor, commands);

	if (!token) return null;

	return { args: token.args, name: token.name, start: token.start };
}

export function isCommandDismissed(
	snapshot: CommandDismissSnapshot | null,
	token: CommandToken
): boolean {
	return (
		!!snapshot &&
		snapshot.start === token.start &&
		snapshot.name === token.name &&
		snapshot.args === token.args
	);
}

/** The input with `[start, end)` taken out, and where the caret goes: where it was. */
export function removeCommandToken(
	value: string,
	token: { start: number; end: number }
): { value: string; caret: number } {
	let before = value.slice(0, token.start);
	let after = value.slice(token.end);

	// one space is left where the token was, none at the start or end of a line
	if ((!before || isWhitespace(before.at(-1))) && after.startsWith(SPACE)) {
		after = after.slice(SPACE.length);
	} else if (before.endsWith(SPACE) && (!after || isWhitespace(after[0]))) {
		before = before.slice(0, -SPACE.length);
	}

	return { caret: before.length, value: before + after };
}

/** The input with `/cwd <query>` in `[start, end)`, and the offset past the query. */
export function replaceWithCwdCommand(
	value: string,
	token: { start: number; end: number },
	query: string
): { value: string; start: number; caret: number } {
	const text = `${CWD_COMMAND_PREFIX}${query}`;

	return {
		caret: token.start + text.length,
		start: token.start,
		value: value.slice(0, token.start) + text + value.slice(token.end)
	};
}

/**
 * The input with `/cwd <query>` written at `caret`, set apart from the words around it by
 * a space, and where the command starts and the query ends.
 */
export function insertCwdCommand(
	value: string,
	caret: number,
	query: string
): { value: string; start: number; caret: number } {
	const before = value.slice(0, caret);
	const after = value.slice(caret);
	const lead = before && !isWhitespace(before.at(-1)) ? SPACE : '';
	const trail = after && !isWhitespace(after[0]) ? SPACE : '';
	const start = before.length + lead.length;

	return replaceWithCwdCommand(before + lead + trail + after, { end: start, start }, query);
}

/**
 * Finds the `/cwd` command again after an edit somewhere else: at `start` if the edit was
 * after it, or moved by `shift` (the change in the input's length) if it was before. Its
 * argument is `query` when the text still reads so, or else runs to two spaces or the end
 * of the line. Null when the command is gone, even if the message names `/cwd` elsewhere.
 */
export function locateCwdCommand(
	value: string,
	start: number,
	query: string,
	shift = 0
): { start: number; end: number; query: string } | null {
	const isCwdAt = (index: number) =>
		startsCommand(value, index) &&
		value.startsWith(CWD_COMMAND_PREFIX.trimEnd(), index) &&
		(index + CWD_COMMAND_PREFIX.length - 1 === value.length ||
			isWhitespace(value[index + CWD_COMMAND_PREFIX.length - 1]));
	const found = [start, start + shift].find(isCwdAt);

	if (found === undefined) return null;

	const argsStart = Math.min(found + CWD_COMMAND_PREFIX.length, value.length);

	if (value.startsWith(query, argsStart)) {
		return { end: argsStart + query.length, query, start: found };
	}

	let end = argsStart;

	while (
		end < value.length &&
		(!isWhitespace(value[end]) || (value[end] === SPACE && value[end + 1] !== SPACE))
	) {
		end++;
	}

	return { end, query: value.slice(argsStart, end).trimEnd(), start: found };
}
