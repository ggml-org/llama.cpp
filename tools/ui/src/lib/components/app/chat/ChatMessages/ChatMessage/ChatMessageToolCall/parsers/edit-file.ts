// Meta parser for `edit_file` tool calls. Reads the file path and the
// array of edits from the streamed args (partial JSON for incremental
// rendering), plus the result blob for `result` / `edits_applied` /
// `error` fields.

import { BuiltInTool } from '$lib/enums';
import { FILE_PATH_SEPARATOR_REGEX } from '$lib/constants';
import { tryParseToolResultObject, type AgenticSection } from '$lib/utils';
import { parseToolArgs } from './_shared';

export type EditFileEdit = {
	oldText: string;
	newText: string;
};

export type EditFileMeta = {
	fileName: string;
	filePath: string;
	edits: EditFileEdit[];
	resultMessage?: string;
	editsApplied?: number;
	errorMessage?: string;
};

export function parseEditFileMeta(section: AgenticSection): EditFileMeta | null {
	const args = parseToolArgs(BuiltInTool.EDIT_FILE, section, { partial: true });
	if (!args) return null;

	const rawPath = args.path ?? args.file_path ?? args.filePath;
	if (typeof rawPath !== 'string' || !rawPath) return null;

	const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;

	// Filter the streamed edits array strictly: each entry must be an
	// object with a non-empty `old_text`. Edits without an old_text
	// would diff against empty and render as a full re-write.
	const rawEdits = Array.isArray(args.edits) ? args.edits : [];
	const edits: EditFileEdit[] = [];
	for (const e of rawEdits) {
		if (!e || typeof e !== 'object' || Array.isArray(e)) continue;
		const obj = e as Record<string, unknown>;
		const oldText = typeof obj.old_text === 'string' ? obj.old_text : '';
		if (!oldText) continue;
		const newText = typeof obj.new_text === 'string' ? obj.new_text : '';
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
		fileName,
		filePath: rawPath,
		edits,
		resultMessage,
		editsApplied,
		errorMessage
	};
}
