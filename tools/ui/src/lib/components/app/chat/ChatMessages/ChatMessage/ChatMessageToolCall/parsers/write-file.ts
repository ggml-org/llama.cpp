// Meta parser for `write_file` tool calls. Reads the path/content from
// the streamed args (partial JSON so we can render before the call
// finishes) and surfaces `bytes`, `result`, and `error` from the
// result blob.

import { BuiltInTool } from '$lib/enums';
import {
	DEFAULT_LANGUAGE,
	FILE_PATH_SEPARATOR_REGEX,
	TEXT_LANGUAGE_PREFIX_REGEX
} from '$lib/constants';
import { getFileTypeByExtension, tryParseToolResultObject, type AgenticSection } from '$lib/utils';
import { parseToolArgs } from './_shared';

export type WriteFileMeta = {
	fileName: string;
	filePath: string;
	language: string;
	content: string;
	bytesWritten?: number;
	resultMessage?: string;
	errorMessage?: string;
};

export function parseWriteFileMeta(section: AgenticSection): WriteFileMeta | null {
	const args = parseToolArgs(BuiltInTool.WRITE_FILE, section, { partial: true });
	if (!args) return null;

	// Tool contracts drifted over time: some models emit `path`,
	// others `file_path` / `filePath`. Accept all three.
	const rawPath = args.path ?? args.file_path ?? args.filePath;
	if (typeof rawPath !== 'string' || !rawPath) return null;

	const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;
	const content = typeof args.content === 'string' ? args.content : '';
	const language =
		getFileTypeByExtension(rawPath)?.replace(TEXT_LANGUAGE_PREFIX_REGEX, '') ?? DEFAULT_LANGUAGE;

	const resultObj = tryParseToolResultObject(section.toolResult);
	const bytesWritten =
		resultObj && Number.isFinite(Number(resultObj.bytes)) ? Number(resultObj.bytes) : undefined;
	const resultMessage = typeof resultObj?.result === 'string' ? resultObj.result : undefined;
	const errorMessage = typeof resultObj?.error === 'string' ? resultObj.error : undefined;

	return {
		fileName,
		filePath: rawPath,
		language,
		content,
		bytesWritten,
		resultMessage,
		errorMessage
	};
}
