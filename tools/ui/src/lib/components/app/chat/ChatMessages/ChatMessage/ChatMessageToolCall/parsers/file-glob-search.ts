// Meta parser for `file_glob_search` tool calls. Reads the path,
// include pattern, and optional exclude from the args (strict parsing)
// and the matches from the result blob. Like grep_search, the result
// parser keeps the original raw-text fallback for MCP servers that
// emit unparseable output.

import { BuiltInTool } from '$lib/enums';
import { splitSearchSummaryList, type AgenticSection } from '$lib/utils';
import { parseToolArgs } from './_shared';

export type FileGlobSearchMeta = {
	path: string;
	include: string;
	exclude?: string;
	matches: string[];
	totalMatches?: number;
	errorMessage?: string;
};

export function parseFileGlobSearchMeta(section: AgenticSection): FileGlobSearchMeta | null {
	const args = parseToolArgs(BuiltInTool.FILE_GLOB_SEARCH, section);
	if (!args) return null;

	const path = typeof args.path === 'string' ? args.path : '';
	const include = typeof args.include === 'string' && args.include ? args.include : '**';
	const exclude = typeof args.exclude === 'string' && args.exclude ? args.exclude : undefined;
	if (!path) return null;

	let matches: string[] = [];
	let totalMatches: number | undefined;
	let errorMessage: string | undefined;

	const toolResultString = section.toolResult;
	if (toolResultString) {
		try {
			const parsed: unknown = JSON.parse(toolResultString);
			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				const obj = parsed as Record<string, unknown>;
				if (typeof obj.error === 'string') {
					errorMessage = obj.error;
				} else if (typeof obj.plain_text_response === 'string') {
					const split = splitSearchSummaryList(obj.plain_text_response, (total) => {
						totalMatches = total;
					});
					matches = split.lines;
				}
			}
		} catch {
			// See grep-search.ts: same fallback used there.
			const split = splitSearchSummaryList(toolResultString, (total) => {
				totalMatches = total;
			});
			matches = split.lines;
		}
	}

	return { path, include, exclude, matches, totalMatches, errorMessage };
}
