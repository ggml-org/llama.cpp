// Meta parser for `edit_file` tool calls. Reads the file path and the
// array of edits from the streamed args (partial JSON for incremental
// rendering), plus the result blob for `result` / `edits_applied` /
// `error` fields.

import { BuiltInTool } from '$lib/enums';
import { tryParseToolResultObject, type AgenticSection } from '$lib/utils';
import { truncatedArgKey } from '$lib/utils/parse-partial-json-args';
import { parseToolArgs } from './_shared';

export type EditFileEdit = {
	oldText: string;
	newText: string;
};

export type EditFileMeta = {
	filePath: string;
	edits: EditFileEdit[];
	resultMessage?: string;
	editsApplied?: number;
	errorMessage?: string;
};

export function parseEditFileMeta(section: AgenticSection): EditFileMeta | null {
	const args = parseToolArgs(BuiltInTool.EDIT_FILE, section, { partial: true });
	if (!args) return null;

	const pathKey =
		args.path != null ? 'path' : args.file_path != null ? 'file_path' : ('filePath' as const);
	const rawPath = args[pathKey];
	if (typeof rawPath !== 'string' || !rawPath) return null;

	// Filter the streamed edits array strictly: each entry must be an
	// object with a non-empty `old_text`. Edits without an old_text
	// would diff against empty and render as a full re-write.
	//
	// The same applies in reverse, and it is the common case while streaming:
	// `old_text` arrives in full before `new_text` begins, so an edit rendered in
	// that window diffs a complete old value against an empty new one and shows
	// every line as deleted, then snaps once the replacement lands. Hold the edit
	// back until `new_text` has actual content. An empty `new_text` mid-stream is
	// ambiguous - it reads the same whether the replacement is a genuine deletion
	// or simply has not arrived - so resolve it by waiting: once the args close, a
	// real deletion (`new_text: ""`) renders normally.
	const argsComplete = truncatedArgKey(section.toolArgs ?? '') === null;
	const rawEdits = Array.isArray(args.edits) ? args.edits : [];
	const edits: EditFileEdit[] = [];
	for (const e of rawEdits) {
		if (!e || typeof e !== 'object' || Array.isArray(e)) continue;
		const obj = e as Record<string, unknown>;
		const oldText = typeof obj.old_text === 'string' ? obj.old_text : '';
		if (!oldText) continue;
		const streamedNewText = typeof obj.new_text === 'string' ? obj.new_text : '';
		if (!argsComplete && streamedNewText.length === 0) continue;
		const newText = streamedNewText;
		edits.push({ oldText, newText });
	}

	const resultObj = tryParseToolResultObject(section.toolResult);
	let resultMessage: string | undefined;
	let editsApplied: number | undefined;
	let errorMessage: string | undefined;
	if (typeof resultObj?.error === 'string') {
		errorMessage = resultObj.error;
	} else if (resultObj) {
		if (typeof resultObj.result === 'string') {
			resultMessage = resultObj.result;
		}
		if (Number.isFinite(Number(resultObj.edits_applied))) {
			editsApplied = Number(resultObj.edits_applied);
		}
	}

	return {
		filePath: rawPath,
		edits,
		resultMessage,
		editsApplied,
		errorMessage
	};
}
