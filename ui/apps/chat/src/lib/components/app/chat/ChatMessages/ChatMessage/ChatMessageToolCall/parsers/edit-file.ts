// Meta parser for `edit_file` tool calls. Reads the file path and the
// array of edits from the streamed args (partial JSON for incremental
// rendering), plus the result blob for `result` / `edits_applied` /
// `error` fields. The path is display-only (block title), so a call
// that has not streamed one still shows its diffs.

import {
	extractToolArgPath,
	extractToolArgString,
	fileNameFromPath,
	parseToolArgs
} from './_shared';
import { TOOL_ARG_PATH_KEYS } from '#lib/constants/index.ts';
import { BuiltInTool, DiffLineKind } from '#lib/enums/index.ts';
import type {
	AgenticSection,
	EditFileEdit,
	EditFileMeta,
	EditFileTitleMeta
} from '#lib/types/index.ts';
import { computeLineDiff, memoizeBounded, tryParseToolResultObject } from '#lib/utils/index.ts';

export function parseEditFileMeta(section: AgenticSection): EditFileMeta | null {
	const args = parseToolArgs(BuiltInTool.SERVER_EDIT_FILE, section, { partial: true });

	if (!args) return null;

	const filePath = extractToolArgPath(args);
	const fileName = fileNameFromPath(filePath);
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
		filePath,
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

	let filePath = extractToolArgString(section.toolArgs, TOOL_ARG_PATH_KEYS);

	if (!filePath) {
		filePath = extractToolArgPath(
			parseToolArgs(BuiltInTool.SERVER_EDIT_FILE, section, { partial: true })
		);
	}

	const fileName = fileNameFromPath(filePath);
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

	return { editsApplied, errorMessage, fileName, filePath, resultMessage };
}

/**
 * Lines added and removed by a finished `edit_file` call, or null when it
 * failed or changed nothing. Memoized per call: the title that shows it
 * renders for every row, while sections are rebuilt on every streamed chunk.
 * The id names the call, so a call without one is computed uncached; the
 * sizes guard against an id that another conversation reuses.
 */
export const editFileDiffStats = memoizeBounded(
	buildEditFileDiffStats,
	(section) =>
		section.toolCallId
			? `${section.toolCallId}:${section.toolArgs?.length ?? 0}:${section.toolResult?.length ?? 0}`
			: null,
	64
);

function buildEditFileDiffStats(
	section: AgenticSection
): { added: number; removed: number } | null {
	const meta = parseEditFileMeta(section);

	if (!meta || meta.errorMessage || meta.edits.length === 0) return null;

	const stats = { added: 0, removed: 0 };

	for (const edit of meta.edits) {
		for (const line of computeLineDiff(edit.oldText, edit.newText)) {
			if (line.kind === DiffLineKind.ADD) stats.added++;
			else if (line.kind === DiffLineKind.REMOVE) stats.removed++;
		}
	}

	return stats;
}
