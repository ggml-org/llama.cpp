// Meta parser for `write_file` tool calls. Reads the path/content from
// the streamed args (partial JSON so we can render before the call
// finishes) and surfaces `bytes`, `result`, and `error` from the
// result blob. The path is display-only (title + highlight language),
// so a call that has not streamed one still shows its content.

import {
	extractToolArgPath,
	extractToolArgString,
	fileNameFromPath,
	parseToolArgs
} from './_shared';
import { CODE_BLOCK, TOOL_ARG_PATH_KEYS } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection, WriteFileMeta, WriteFileTitleMeta } from '$lib/types';
import { getFileTypeByExtension, tryParseToolResultObject } from '$lib/utils';

// Plain text when the call carries no path, or one with no known extension.
function languageForPath(filePath: string | undefined): string {
	if (!filePath) return CODE_BLOCK.DEFAULT_LANGUAGE;

	return (
		getFileTypeByExtension(filePath)?.replace(CODE_BLOCK.TEXT_LANGUAGE_PREFIX_REGEX, '') ??
		CODE_BLOCK.DEFAULT_LANGUAGE
	);
}

export function parseWriteFileMeta(section: AgenticSection): WriteFileMeta | null {
	const args = parseToolArgs(BuiltInTool.SERVER_WRITE_FILE, section, { partial: true });

	if (!args) return null;

	const filePath = extractToolArgPath(args);
	const fileName = fileNameFromPath(filePath);
	// undefined, not '', so an intentionally empty write still renders
	const content = typeof args.content === 'string' ? args.content : undefined;
	const language = languageForPath(filePath);
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
		filePath,
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

	let filePath = extractToolArgString(section.toolArgs, TOOL_ARG_PATH_KEYS);

	if (!filePath) {
		filePath = extractToolArgPath(
			parseToolArgs(BuiltInTool.SERVER_WRITE_FILE, section, { partial: true })
		);
	}

	const resultObj = tryParseToolResultObject(section.toolResult);
	const bytesWritten =
		resultObj && Number.isFinite(Number(resultObj.bytes)) ? Number(resultObj.bytes) : undefined;
	const resultMessage = typeof resultObj?.result === 'string' ? resultObj.result : undefined;
	const errorMessage = typeof resultObj?.error === 'string' ? resultObj.error : undefined;

	return {
		bytesWritten,
		errorMessage,
		fileName: fileNameFromPath(filePath),
		filePath,
		language: languageForPath(filePath),
		resultMessage
	};
}
