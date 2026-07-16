// Meta parser for `read_file` tool calls. Reads the file path and an
// optional line range (either `start_line`+`end_line` or
// `start_line`+`line_count`). Args are parsed partially so a header
// can render incrementally as the file path streams in.

import { BuiltInTool } from '$lib/enums';
import {
	DEFAULT_LANGUAGE,
	FILE_PATH_SEPARATOR_REGEX,
	TEXT_LANGUAGE_PREFIX_REGEX
} from '$lib/constants';
import { getFileTypeByExtension, type AgenticSection } from '$lib/utils';
import { parseToolArgs } from './_shared';

export type ReadFileMeta = {
	fileName: string;
	lineRange: { start: number; end: number } | null;
	language: string;
};

export function parseReadFileMeta(section: AgenticSection): ReadFileMeta | null {
	const args = parseToolArgs(BuiltInTool.READ_FILE, section, { partial: true });
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
		lineRange = { start: sNum, end: eNum };
	} else if (startRaw != null && countRaw != null) {
		const cNum = Number(countRaw);
		if (Number.isFinite(sNum) && Number.isFinite(cNum)) {
			lineRange = { start: sNum, end: sNum + cNum - 1 };
		}
	}

	const fileType = getFileTypeByExtension(fileName);
	const language = fileType ? fileType.replace(TEXT_LANGUAGE_PREFIX_REGEX, '') : DEFAULT_LANGUAGE;

	return { fileName, lineRange, language };
}
