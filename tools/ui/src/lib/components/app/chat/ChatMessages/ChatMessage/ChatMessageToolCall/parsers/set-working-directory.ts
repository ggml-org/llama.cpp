// Meta parser for `set_working_directory` tool calls. Reads the `path`
// argument from the tool call args and surfaces any error from the
// result blob. The result is a plain string (not JSON) on success,
// so we fall back to scanning raw lines for the `Error:` prefix.

import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/utils';
import { parseToolArgs } from './_shared';

export type SetWorkingDirectoryMeta = {
	path: string | null;
	errorMessage?: string;
};

export function parseSetWorkingDirectoryMeta(section: AgenticSection): SetWorkingDirectoryMeta | null {
	const args = parseToolArgs(BuiltInTool.SET_WORKING_DIRECTORY, section);
	if (!args) return null;

	const rawPath = args.path;
	const path = typeof rawPath === 'string' ? rawPath.trim() || null : null;

	let errorMessage: string | undefined;
	const toolResultString = section.toolResult;
	if (toolResultString) {
		// Try JSON first — the service may wrap the result.
		let parsedObject: Record<string, unknown> | null = null;
		try {
			const parsed: unknown = JSON.parse(toolResultString);
			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				parsedObject = parsed as Record<string, unknown>;
			}
		} catch {
			parsedObject = null;
		}
		if (typeof parsedObject?.error === 'string') {
			errorMessage = parsedObject.error;
		} else if (typeof parsedObject?.content === 'string' && parsedObject.content.startsWith('Error:')) {
			errorMessage = parsedObject.content.slice('Error:'.length).trim();
		} else if (!parsedObject) {
			// Not JSON — scan raw lines for the `Error:` prefix.
			const errorLine = toolResultString
				.split('\n')
				.map((line) => line.trim())
				.find((line) => line.startsWith('Error:'));
			if (errorLine) errorMessage = errorLine.slice('Error:'.length).trim();
		}
	}

	return { path, errorMessage };
}
