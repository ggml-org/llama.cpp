// Meta parser for `grep_search` tool calls. Reads the path/pattern
// triplet from args (strict parsing - we wait for the args to
// complete) and the matches from the result blob. The result parser
// keeps the original "scan result as raw text on JSON.parse failure"
// fallback so MCP servers that return unparseable output still get
// surfaced.

import { BuiltInTool } from '$lib/enums';
import { splitSearchSummaryList, type AgenticSection } from '$lib/utils';
import { parseToolArgs } from './_shared';

export type GrepSearchMatch = {
	file: string;
	line?: number;
	content: string;
};

export type GrepSearchMeta = {
	path: string;
	pattern: string;
	include: string;
	exclude?: string;
	showLineNumbers: boolean;
	matches: GrepSearchMatch[];
	totalMatches?: number;
	errorMessage?: string;
};

export function parseGrepSearchMeta(section: AgenticSection): GrepSearchMeta | null {
	const args = parseToolArgs(BuiltInTool.GREP_SEARCH, section);
	if (!args) return null;

	const path = typeof args.path === 'string' ? args.path : '';
	const pattern = typeof args.pattern === 'string' ? args.pattern : '';
	if (!path || !pattern) return null;

	const include = typeof args.include === 'string' && args.include ? args.include : '**';
	const exclude = typeof args.exclude === 'string' && args.exclude ? args.exclude : undefined;
	const showLineNumbers = args.return_line_numbers === true;

	let matches: GrepSearchMatch[] = [];
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
					matches = split.lines.map((line) => parseGrepLine(line, showLineNumbers));
				}
			}
		} catch {
			// Result wasn't JSON: keep behaviour for MCP servers that
			// emit raw text and treat each line as a `<file>:<content>`
			// (or `<file>:<line>:<content>`) match.
			const split = splitSearchSummaryList(toolResultString, (total) => {
				totalMatches = total;
			});
			matches = split.lines.map((line) => parseGrepLine(line, showLineNumbers));
		}
	}

	return {
		path,
		pattern,
		include,
		exclude,
		showLineNumbers,
		matches,
		totalMatches,
		errorMessage
	};
}

function parseGrepLine(line: string, showLineNumbers: boolean): GrepSearchMatch {
	// Server output:
	//   <file>:<content>          when return_line_numbers=false
	//   <file>:<lineno>:<content> when return_line_numbers=true
	const firstColon = line.indexOf(':');
	if (firstColon === -1) {
		return { file: line, content: '' };
	}
	const file = line.slice(0, firstColon);
	const tail = line.slice(firstColon + 1);

	if (!showLineNumbers) {
		return { file, content: tail };
	}

	const secondColon = tail.indexOf(':');
	if (secondColon === -1) {
		return { file, content: tail };
	}
	const lineNum = parseInt(tail.slice(0, secondColon), 10);
	return {
		file,
		line: Number.isFinite(lineNum) ? lineNum : undefined,
		content: tail.slice(secondColon + 1)
	};
}
