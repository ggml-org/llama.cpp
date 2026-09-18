// Meta parser for `read_file` tool calls. Reads the file path and an
// optional line range (either `start_line`+`end_line` or
// `start_line`+`line_count`). Args are parsed partially so a header
// can render incrementally as the file path streams in.

import { parseToolArgs } from './_shared';
import { CODE_BLOCK, FILE_PATH_SEPARATOR_REGEX } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types';
import { getFileTypeByExtension } from '$lib/utils';

export type ReadFileMeta = {
	fileName: string;
	lineRange: { start: number; end: number } | null;
	language: string;
};

export function parseReadFileMeta(section: AgenticSection): ReadFileMeta | null {
	const args = parseToolArgs(BuiltInTool.SERVER_READ_FILE, section, { partial: true });

	if (!args) return null;

	const rawPath = args.path ?? args.file_path ?? args.filePath;

	if (typeof rawPath !== 'string' || !rawPath) return null;

	const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;
	// Models emit range arguments under several aliases. Accept all to
	// stay forgiving across prompt variations.
	const startRaw = args.start_line ?? args.line_start ?? args.startLine ?? args.from_line;
	const endRaw = args.end_line ?? args.line_end ?? args.endLine ?? args.to_line;
	const countRaw = args.line_count ?? args.count ?? args.num_lines;

	let lineRange: { start: number; end: number } | null = null;

	const sNum = Number(startRaw);
	const eNum = Number(endRaw);

	if (startRaw != null && endRaw != null && Number.isFinite(sNum) && Number.isFinite(eNum)) {
		lineRange = { end: eNum, start: sNum };
	} else if (startRaw != null && countRaw != null) {
		const cNum = Number(countRaw);

		if (Number.isFinite(sNum) && Number.isFinite(cNum)) {
			lineRange = { end: sNum + cNum - 1, start: sNum };
		}
	}

	const fileType = getFileTypeByExtension(fileName);
	const language = fileType
		? fileType.replace(CODE_BLOCK.TEXT_LANGUAGE_PREFIX_REGEX, '')
		: CODE_BLOCK.DEFAULT_LANGUAGE;

	return { fileName, language, lineRange };
}
