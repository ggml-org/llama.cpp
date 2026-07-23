// Meta parser for `run_javascript` tool calls. Reads the JS code and
// optional timeout from args (strict parsing) and surfaces any error
// from the result blob. SandboxService.formatReply emits a JSON object
// containing an `error` field on failure, but a partial/non-JSON
// failure renders as a flat line beginning with `Error:`. Both shapes
// are handled.

import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/utils';
import { parseToolArgs } from './_shared';

export type RunJavascriptMeta = {
	code: string;
	timeoutMs?: number;
	errorMessage?: string;
};

export function parseRunJavascriptMeta(section: AgenticSection): RunJavascriptMeta | null {
	const args = parseToolArgs(BuiltInTool.RUN_JAVASCRIPT, section);
	if (!args) return null;

	const code = typeof args.code === 'string' ? args.code : '';
	if (!code) return null;

	const timeoutRaw = Number(args.timeout_ms);
	const timeoutMs = Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? timeoutRaw : undefined;

	let errorMessage: string | undefined;
	const toolResultString = section.toolResult;
	if (toolResultString) {
		// Branches matter here: a JSON object can carry `error`, but a
		// JSON array always represents successful output (sandbox returns
		// the array of values). Only when the result isn't a JSON object
		// do we scan raw lines for the `Error:` prefix.
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
		} else if (!parsedObject) {
			const errorLine = toolResultString
				.split('\n')
				.map((line) => line.trim())
				.find((line) => line.startsWith('Error:'));
			if (errorLine) errorMessage = errorLine.slice('Error:'.length).trim();
		}
	}

	return { code, timeoutMs, errorMessage };
}
