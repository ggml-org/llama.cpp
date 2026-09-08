// Meta parser for `write_file` tool calls. Reads the path/content from
// the streamed args (partial JSON so we can render before the call
// finishes) and surfaces `bytes`, `result`, and `error` from the
// result blob.

import { extractToolArgString, parseToolArgs } from './_shared';
import { CODE_BLOCK, FILE_PATH_SEPARATOR_REGEX, TOOL_ARG_PATH_KEYS } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection, WriteFileMeta, WriteFileTitleMeta } from '$lib/types';
import { getFileTypeByExtension, tryParseToolResultObject } from '$lib/utils';

export function parseWriteFileMeta(section: AgenticSection): WriteFileMeta | null {
	const args = parseToolArgs(BuiltInTool.SERVER_WRITE_FILE, section, { partial: true });

	if (!args) return null;

	// Tool contracts drifted over time: some models emit `path`,
	// others `file_path` / `filePath`. Accept all three.
	const rawPath = args.path ?? args.file_path ?? args.filePath;

	if (typeof rawPath !== 'string' || !rawPath) return null;

	const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;
	const content = typeof args.content === 'string' ? args.content : '';
	const language =
		getFileTypeByExtension(rawPath)?.replace(CODE_BLOCK.TEXT_LANGUAGE_PREFIX_REGEX, '') ??
		CODE_BLOCK.DEFAULT_LANGUAGE;
	const resultObj = tryParseToolResultObject(section.toolResult);
	const bytesWritten =
		resultObj && Number.isFinite(Number(resultObj.bytes)) ? Number(resultObj.bytes) : undefined;
	const resultMessage = typeof resultObj?.result === 'string' ? resultObj.result : undefined;
	const errorMessage = typeof resultObj?.error === 'string' ? resultObj.error : undefined;

	return {
		bytesWritten,
		content,
		errorMessage,
		fileName,
		filePath: rawPath,
		language,
		resultMessage
	};
}

/**
 * Title-tier meta for write_file blocks: everything the header and status
 * pill render, obtained without parsing the embedded file content. The path
 * comes from a targeted key extraction; the full parse runs only as a
 * fallback for arg shapes the extraction can't see.
 */
export function parseWriteFileTitleMeta(section: AgenticSection): WriteFileTitleMeta | null {
	if (section.toolName !== BuiltInTool.SERVER_WRITE_FILE || !section.toolArgs) return null;

	let rawPath: string | undefined = extractToolArgString(section.toolArgs, TOOL_ARG_PATH_KEYS);

	if (!rawPath) {
		const args = parseToolArgs(BuiltInTool.SERVER_WRITE_FILE, section, { partial: true });
		const fallbackPath = args?.path ?? args?.file_path ?? args?.filePath;

		if (typeof fallbackPath === 'string' && fallbackPath) rawPath = fallbackPath;
	}

	if (!rawPath) return null;

	const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;
	const language =
		getFileTypeByExtension(rawPath)?.replace(CODE_BLOCK.TEXT_LANGUAGE_PREFIX_REGEX, '') ??
		CODE_BLOCK.DEFAULT_LANGUAGE;
	const resultObj = tryParseToolResultObject(section.toolResult);
	const bytesWritten =
		resultObj && Number.isFinite(Number(resultObj.bytes)) ? Number(resultObj.bytes) : undefined;
	const resultMessage = typeof resultObj?.result === 'string' ? resultObj.result : undefined;
	const errorMessage = typeof resultObj?.error === 'string' ? resultObj.error : undefined;

	return {
		bytesWritten,
		errorMessage,
		fileName,
		filePath: rawPath,
		language,
		resultMessage
	};
}
