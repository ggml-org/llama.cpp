// Meta parser for terminal tool calls (`exec_shell_command`, `run_bash`). Surfaces the
// command text from args `command` / `cmd` / `shell_command` aliases.
// Args are parsed partially so the title can render the command while
// it is still streaming.
// The exit-status and error parsing live in their own utilities
// (`parse-exec-shell-status.ts` / `parse-exec-shell-error.ts`) - this
// file only deals with what's strictly about *calling* the tool, since
// the error / exit status elide from call-section to result-section.

import { parseToolArgs } from './_shared';
import { TERMINAL_TOOLS } from '#lib/constants/index.ts';
import type { BuiltInTool } from '#lib/enums/index.ts';
import type { AgenticSection } from '#lib/types/index.ts';

export type ExecShellCommandMeta = {
	command: string;
};

export function parseExecShellCommandMeta(section: AgenticSection): ExecShellCommandMeta | null {
	if (!section.toolName || !TERMINAL_TOOLS.has(section.toolName)) return null;

	const args = parseToolArgs(section.toolName as BuiltInTool, section, { partial: true });

	if (!args) return null;

	const commandRaw = args.command ?? args.cmd ?? args.shell_command;

	if (typeof commandRaw !== 'string' || !commandRaw) return null;

	return { command: commandRaw };
}
