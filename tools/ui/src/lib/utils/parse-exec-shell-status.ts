/**
 * Parsing helpers for `exec_shell_command` tool output.
 *
 * The server appends one final line to the response - an exit-code summary
 * shaped as `[exit code: N]` (and optionally followed by `[exit due to timed
 * out]`) - so the renderer can color that final line based on success/failure
 * without parsing the entire output stream.
 */

export interface ExecShellExitStatus {
	code: number;
	timedOut: boolean;
	/** Length-prefix slice for matching against the rendered lines list. */
	rawText: string;
}

// Anchor to the absolute end so intermediate "[exit code: N]" string content
// (e.g. a shell echo) doesn't false-positive.
const EXIT_CODE_TAIL_REGEX = /\[exit code: (-?\d+)\](?: \[exit due to timed out\])?\s*$/;

export function parseExecShellCommandExitStatus(
	toolResultString: string | undefined
): ExecShellExitStatus | undefined {
	if (!toolResultString) return undefined;

	const match = toolResultString.match(EXIT_CODE_TAIL_REGEX);
	if (!match) return undefined;

	return {
		code: Number.parseInt(match[1], 10),
		timedOut: match[0].includes('exit due to timed out'),
		rawText: match[0]
	};
}

/**
 * Returns true when the supplied rendered line equals (trimmed) the
 * trailing exit-code text. Used by the renderer to drop the duplicated
 * representation (since the trailing line is replaced by a status badge).
 */
export function isExitCodeSummaryLine(
	lineText: string,
	status: ExecShellExitStatus | undefined
): boolean {
	if (!status) return false;
	return lineText.trim() === status.rawText.trim();
}
