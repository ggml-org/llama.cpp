// Meta parser for `edit_file` tool calls. Reads the file path and the
// array of edits from the streamed args (partial JSON for incremental
// rendering), plus the result blob for `result` / `edits_applied` /
// `error` fields.

import { extractToolArgString, parseToolArgs } from './_shared';
import { FILE_PATH_SEPARATOR_REGEX, TOOL_ARG_PATH_KEYS } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection, EditFileEdit, EditFileMeta, EditFileTitleMeta } from '$lib/types';
import { tryParseToolResultObject } from '$lib/utils';

export function parseEditFileMeta(section: AgenticSection): EditFileMeta | null {
	const args = parseToolArgs(BuiltInTool.SERVER_EDIT_FILE, section, { partial: true });

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

		edits.push({ newText, oldText });
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
		edits,
		editsApplied,
		errorMessage,
		fileName,
		filePath: rawPath,
		resultMessage
	};
}

/**
 * Title-tier meta for edit_file blocks: everything the header and status
 * pill render, obtained without parsing the embedded edit strings. The path
 * comes from a targeted key extraction; the full parse runs only as a
 * fallback for arg shapes the extraction can't see.
 */
export function parseEditFileTitleMeta(section: AgenticSection): EditFileTitleMeta | null {
	if (section.toolName !== BuiltInTool.SERVER_EDIT_FILE || !section.toolArgs) return null;

	let rawPath: string | undefined = extractToolArgString(section.toolArgs, TOOL_ARG_PATH_KEYS);

	if (!rawPath) {
		const args = parseToolArgs(BuiltInTool.SERVER_EDIT_FILE, section, { partial: true });
		const fallbackPath = args?.path ?? args?.file_path ?? args?.filePath;

		if (typeof fallbackPath === 'string' && fallbackPath) rawPath = fallbackPath;
	}

	if (!rawPath) return null;

	const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;
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

	return { editsApplied, errorMessage, fileName, filePath: rawPath, resultMessage };
}
