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
// (e.g. a shell echo) doesn't false-positive. The marker is at most ~50 chars
// with the timed-out suffix, so matching a tail slice keeps the cost constant
// for megabyte exec outputs instead of scanning the whole blob.
const EXIT_CODE_TAIL_REGEX = /\[exit code: (-?\d+)\](?: \[exit due to timed out\])?\s*$/;
const EXIT_CODE_TAIL_SCAN = 128;

export function parseExecShellCommandExitStatus(
	toolResultString: string | undefined
): ExecShellExitStatus | undefined {
	if (!toolResultString) return undefined;

	const match = toolResultString.slice(-EXIT_CODE_TAIL_SCAN).match(EXIT_CODE_TAIL_REGEX);

	if (!match) return undefined;

	return {
		code: Number.parseInt(match[1], 10),
		rawText: match[0],
		timedOut: match[0].includes('exit due to timed out')
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
