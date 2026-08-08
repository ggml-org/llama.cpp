// Meta parser for `read_file` tool calls. Reads the file path and an
// optional line range (either `start_line`+`end_line` or
// `start_line`+`line_count`). Args are parsed partially so a header
// can render incrementally as the file path streams in.

import { BuiltInTool } from '$lib/enums';
import { DEFAULT_LANGUAGE, TEXT_LANGUAGE_PREFIX_REGEX } from '$lib/constants';
import { getFileTypeByExtension, type AgenticSection } from '$lib/utils';
import { truncatedArgKey } from '$lib/utils/parse-partial-json-args';
import { parseToolArgs } from './_shared';

/** Every alias the range arguments arrive under, for the in-flight check below. */
const RANGE_ARG_KEYS = new Set([
	'start_line',
	'line_start',
	'startLine',
	'from_line',
	'end_line',
	'line_end',
	'endLine',
	'to_line',
	'line_count',
	'count',
	'num_lines'
]);

export type ReadFileMeta = {
	filePath: string;
	lineRange: { start: number; end: number } | null;
	language: string;
};

export function parseReadFileMeta(section: AgenticSection): ReadFileMeta | null {
	const args = parseToolArgs(BuiltInTool.READ_FILE, section, { partial: true });
	if (!args) return null;

	const pathKey =
		args.path != null ? 'path' : args.file_path != null ? 'file_path' : ('filePath' as const);
	const rawPath = args[pathKey];
	if (typeof rawPath !== 'string' || !rawPath) return null;

	const truncatedKey = truncatedArgKey(section.toolArgs ?? '');

	// The path is shown whole, matching write_file and edit_file. Collapsing it to
	// a basename used to make the header walk through every segment as the value
	// streamed in ("/Us" -> "Us", "/Users/za" -> "za", ...); rendering the raw
	// value is monotonic, so there is nothing to guard against.
	const pathComplete = truncatedKey !== pathKey;

	// Models emit range arguments under several aliases. Accept all to
	// stay forgiving across prompt variations.
	const startRaw = args.start_line ?? args.line_start ?? args.startLine ?? args.from_line;
	const endRaw = args.end_line ?? args.line_end ?? args.endLine ?? args.to_line;
	const countRaw = args.line_count ?? args.count ?? args.num_lines;

	// Numbers stream digit by digit, so a range built from the in-flight value
	// renders "1-4" and then "1-40". Withhold it until the value is closed.
	const rangeStillStreaming = truncatedKey !== null && RANGE_ARG_KEYS.has(truncatedKey);

	let lineRange: { start: number; end: number } | null = null;
	const sNum = Number(startRaw);
	const eNum = Number(endRaw);
	if (rangeStillStreaming) {
		lineRange = null;
	} else if (startRaw != null && endRaw != null && Number.isFinite(sNum) && Number.isFinite(eNum)) {
		lineRange = { start: sNum, end: eNum };
	} else if (startRaw != null && countRaw != null) {
		const cNum = Number(countRaw);
		if (Number.isFinite(sNum) && Number.isFinite(cNum)) {
			lineRange = { start: sNum, end: sNum + cNum - 1 };
		}
	}

	// The language does still need the guard: a partial extension resolves to a
	// different grammar every few tokens and re-highlights the whole block.
	const fileType = pathComplete ? getFileTypeByExtension(rawPath) : null;
	const language = fileType ? fileType.replace(TEXT_LANGUAGE_PREFIX_REGEX, '') : DEFAULT_LANGUAGE;

	return { filePath: rawPath, lineRange, language };
}
